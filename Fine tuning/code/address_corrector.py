# =============================================================================
# address_corrector.py  (v2 — Context-aware with skip-if-valid)
# Vietnamese Address Geographic Name Corrector using Gazetteer Fuzzy-Matching
# =============================================================================

import json
import logging
import os
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class AddressCorrector:
    """
    Correct Vietnamese geographic names in address strings using a
    fuzzy-matching gazetteer of official administrative divisions.

    Data source: madnh/hanhchinhvn (General Statistics Office of Vietnam)
    Contains: 63 provinces + ~705 districts + ~10,599 wards/communes

    v2 improvements over v1:
      - Skip-if-valid: never correct a name that already exactly matches
        a gazetteer entry (prevents Đồng Mai → Đồng Nai)
      - Province context: extract trailing province, then restrict
        district/ward matching to that province
      - Ward-level data: can now validate Phường/Xã names

    Usage:
        corrector = AddressCorrector("data/vietnam_gazetteer.json")
        fixed = corrector.correct("Phường Nỹi Am, Quận Người tành, Đà Vưỡng")
    """

    # Vietnamese administrative keywords (case-insensitive matching)
    _KEYWORDS = [
        "Thành phố", "Thành Phố", "TP",
        "Tỉnh",
        "Quận",
        "Huyện",
        "Thị xã", "Thị Xã", "TX",
        "Thị trấn", "Thị Trấn", "TT",
        "Phường",
        "Xã",
    ]

    # Province-level keywords (match against province gazetteer)
    _PROVINCE_KEYWORDS = {"Thành phố", "Thành Phố", "TP", "Tỉnh"}

    # District-level keywords
    _DISTRICT_KEYWORDS = {"Quận", "Huyện", "Thị xã", "Thị Xã", "TX"}

    # Ward-level keywords
    _WARD_KEYWORDS = {"Phường", "Xã", "Thị trấn", "Thị Trấn", "TT"}

    def __init__(
        self,
        gazetteer_path: str,
        max_distance_ratio: float = 0.35,
        min_entity_length: int = 2,
    ):
        """
        Args:
            gazetteer_path: Path to vietnam_gazetteer.json
            max_distance_ratio: Maximum edit_distance/max_len to accept.
                                0.35 = allow up to 35% character changes.
            min_entity_length: Minimum length of extracted entity to attempt matching.
        """
        self.max_distance_ratio = max_distance_ratio
        self.min_entity_length = min_entity_length

        # Flat name sets for exact-match checks (NFC-lowered)
        self._all_names_lower: Set[str] = set()

        # Structured data for fuzzy matching
        self._province_names: List[Tuple[str, str]] = []     # (norm, original)
        self._district_data: List[Dict] = []                  # {norm, name, province}
        self._ward_data: List[Dict] = []                      # {norm, name, district, province}

        # Province -> districts/wards mapping for context-aware matching
        self._province_to_districts: Dict[str, List[Tuple[str, str]]] = {}
        self._province_to_wards: Dict[str, List[Tuple[str, str]]] = {}

        self._loaded = False
        self._load_gazetteer(gazetteer_path)
        self._build_keyword_pattern()

    def _load_gazetteer(self, path: str):
        """Load and index the gazetteer JSON."""
        if not os.path.exists(path):
            logger.warning(f"[AddressCorrector] Gazetteer not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        provinces = data.get("provinces", [])
        districts = data.get("districts", [])
        wards = data.get("wards", [])

        # Index provinces
        for name in provinces:
            norm = unicodedata.normalize("NFC", name).lower()
            self._province_names.append((norm, name))
            self._all_names_lower.add(norm)

        # Index districts with province mapping
        for d in districts:
            norm = unicodedata.normalize("NFC", d["name"]).lower()
            province = d.get("province", "")
            self._district_data.append({"norm": norm, "name": d["name"], "province": province})
            self._all_names_lower.add(norm)

            prov_key = unicodedata.normalize("NFC", province).lower()
            if prov_key not in self._province_to_districts:
                self._province_to_districts[prov_key] = []
            self._province_to_districts[prov_key].append((norm, d["name"]))

        # Index wards with district+province mapping
        for w in wards:
            norm = unicodedata.normalize("NFC", w["name"]).lower()
            district = w.get("district", "")
            province = w.get("province", "")
            self._ward_data.append({
                "norm": norm, "name": w["name"],
                "district": district, "province": province,
            })
            self._all_names_lower.add(norm)

            prov_key = unicodedata.normalize("NFC", province).lower()
            if prov_key not in self._province_to_wards:
                self._province_to_wards[prov_key] = []
            self._province_to_wards[prov_key].append((norm, w["name"]))

        self._loaded = bool(provinces)
        logger.info(
            f"[AddressCorrector] Loaded gazetteer: "
            f"{len(provinces)} provinces, {len(districts)} districts, "
            f"{len(wards)} wards | {len(self._all_names_lower)} unique names"
        )

    def _build_keyword_pattern(self):
        """Build regex pattern to find administrative keywords."""
        sorted_kw = sorted(self._KEYWORDS, key=len, reverse=True)
        escaped = [re.escape(kw) for kw in sorted_kw]
        self._keyword_re = re.compile(
            r"(?:^|(?<=[\s,]))(" + "|".join(escaped) + r")\s+",
            re.IGNORECASE,
        )

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Levenshtein edit distance."""
        if len(s1) < len(s2):
            return AddressCorrector._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
            prev = curr
        return prev[-1]

    def _is_exact_match(self, entity: str) -> bool:
        """
        Check if entity already exactly matches a gazetteer entry.
        If so, it should NOT be corrected — it's already valid.
        """
        norm = unicodedata.normalize("NFC", entity.strip()).lower()
        return norm in self._all_names_lower

    def _find_best_match(
        self,
        entity: str,
        keyword: str,
        province_context: Optional[str] = None,
    ) -> Optional[str]:
        """
        Find the best gazetteer match for an extracted entity.

        Args:
            entity: The geographic name text (e.g., "Người tành")
            keyword: The preceding keyword (e.g., "Quận")
            province_context: Known province name for context-aware matching.

        Returns:
            Best matching official name, or None if no good match.
        """
        entity_norm = unicodedata.normalize("NFC", entity.strip()).lower()

        if len(entity_norm) < self.min_entity_length:
            return None

        # ── SKIP-IF-VALID: Don't correct already-correct names ──
        if entity_norm in self._all_names_lower:
            return None

        kw_stripped = keyword.strip()

        # Determine candidate pool based on keyword + province context
        candidates = self._get_candidates(kw_stripped, province_context)

        best_name = None
        best_dist = float("inf")

        for cand_lower, cand_original in candidates:
            dist = self._edit_distance(entity_norm, cand_lower)
            max_len = max(len(entity_norm), len(cand_lower))

            if max_len == 0:
                continue

            ratio = dist / max_len

            if ratio <= self.max_distance_ratio and dist < best_dist:
                best_dist = dist
                best_name = cand_original

        return best_name

    def _get_candidates(
        self, keyword: str, province_context: Optional[str]
    ) -> List[Tuple[str, str]]:
        """
        Get candidate list based on keyword type and province context.

        When province is known, restrict to entries within that province.
        Falls back to full list if province-scoped search yields nothing.
        """
        prov_key = None
        if province_context:
            prov_key = unicodedata.normalize("NFC", province_context).lower()

        if keyword in self._PROVINCE_KEYWORDS:
            return self._province_names

        elif keyword in self._DISTRICT_KEYWORDS:
            # Try province-scoped first
            if prov_key and prov_key in self._province_to_districts:
                scoped = self._province_to_districts[prov_key]
                if scoped:
                    return scoped
            # Fallback to all districts
            return [(d["norm"], d["name"]) for d in self._district_data]

        elif keyword in self._WARD_KEYWORDS:
            # Try province-scoped first
            if prov_key and prov_key in self._province_to_wards:
                scoped = self._province_to_wards[prov_key]
                if scoped:
                    return scoped
            # Fallback to all wards
            return [(w["norm"], w["name"]) for w in self._ward_data]

        else:
            # Unknown keyword — search all
            return self._province_names + [(d["norm"], d["name"]) for d in self._district_data]

    def _extract_province_context(self, text: str) -> Optional[str]:
        """
        Try to identify the province from the text (usually the last segment).

        Looks at the last comma-separated segment and tries exact/fuzzy match
        against the province list.
        """
        last_comma = text.rfind(",")
        if last_comma == -1:
            return None

        trailing = text[last_comma + 1:].strip()
        if not trailing or len(trailing) < 2:
            return None

        # Strip any keyword prefix from trailing
        kw_match = self._keyword_re.match(trailing)
        if kw_match:
            trailing = trailing[kw_match.end():].strip()

        if not trailing:
            return None

        trailing_norm = unicodedata.normalize("NFC", trailing).lower()

        # Exact match first
        for norm, original in self._province_names:
            if trailing_norm == norm:
                return original

        # Fuzzy match with stricter threshold for province detection
        best_name = None
        best_dist = float("inf")
        for norm, original in self._province_names:
            dist = self._edit_distance(trailing_norm, norm)
            max_len = max(len(trailing_norm), len(norm))
            if max_len > 0 and dist / max_len <= 0.3 and dist < best_dist:
                best_dist = dist
                best_name = original

        return best_name

    def is_address_like(self, text: str) -> bool:
        """Check if text contains Vietnamese address patterns."""
        if not text:
            return False
        return bool(self._keyword_re.search(text))

    def correct(self, text: str) -> str:
        """
        Correct geographic names in an address string.

        Pipeline:
          1. Check if text is address-like
          2. Extract province context from trailing segment
          3. For each keyword + entity pair:
             a. Skip if entity already matches a valid gazetteer name
             b. Fuzzy-match, preferring entries within the detected province
          4. Return corrected text

        Non-address text is returned unchanged.
        """
        if not text or not self._loaded:
            return text

        if not self.is_address_like(text):
            return text

        # Step 1: Detect province context
        province_ctx = self._extract_province_context(text)

        result = text
        corrections = 0

        # Step 2: Find all keyword positions
        matches = list(self._keyword_re.finditer(result))

        # Process from right to left to preserve indices
        for match in reversed(matches):
            keyword = match.group(1)
            entity_start = match.end()

            # Extract entity: text from after keyword to next comma/keyword/end
            remaining = result[entity_start:]

            boundary_re = re.compile(
                r"[,]|(?:" + "|".join(
                    re.escape(kw) for kw in sorted(self._KEYWORDS, key=len, reverse=True)
                ) + r")"
            )
            boundary_match = boundary_re.search(remaining)
            if boundary_match:
                entity_text = remaining[:boundary_match.start()].strip()
            else:
                entity_text = remaining.strip()

            if not entity_text or len(entity_text) < self.min_entity_length:
                continue

            # Try to find a gazetteer match (skips if already valid)
            best = self._find_best_match(entity_text, keyword, province_ctx)
            if best and best != entity_text:
                entity_end = entity_start + len(entity_text)
                result = result[:entity_start] + best + result[entity_end:]
                corrections += 1

        if corrections > 0:
            logger.debug(
                f"[AddressCorrector] {corrections} corrections: "
                f"'{text}' → '{result}'"
            )

        return result

    def correct_trailing_province(self, text: str) -> str:
        """
        Correct the last segment of an address (usually the province name).
        e.g., "..., Đà Vưỡng" → "..., Đà Nẵng"
        """
        if not text or not self._loaded:
            return text

        last_comma = text.rfind(",")
        if last_comma == -1:
            return text

        trailing = text[last_comma + 1:].strip()
        if not trailing or len(trailing) < 2:
            return text

        # Check if trailing text already has a keyword
        if self._keyword_re.match(trailing):
            return text

        # Skip if already valid
        if self._is_exact_match(trailing):
            return text

        # Try matching against provinces
        trailing_norm = unicodedata.normalize("NFC", trailing).lower()
        best_name = None
        best_dist = float("inf")

        for norm, original in self._province_names:
            dist = self._edit_distance(trailing_norm, norm)
            max_len = max(len(trailing_norm), len(norm))
            if max_len > 0 and dist / max_len <= self.max_distance_ratio and dist < best_dist:
                best_dist = dist
                best_name = original

        if best_name and best_name != trailing:
            result = text[:last_comma + 1] + " " + best_name
            logger.debug(f"[AddressCorrector] Trailing province: '{trailing}' → '{best_name}'")
            return result

        return text
