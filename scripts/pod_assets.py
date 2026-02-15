from __future__ import annotations

"""
Helpers for resolving MTGA card metadata and art images from local MTGA installs.

Card names/grpIds/artIds come from Raw_CardDatabase_*.mtga (SQLite).
Card art is stored in UnityFS bundles under MTGA_Data/Downloads/AssetBundle as
<ArtId>_CardArt_<hash>.mtga. If UnityPy is installed, we extract Texture2D from
the bundle and return a data URL (base64 PNG). If UnityPy is missing or no bundle
is found, art_uri returns None.
"""

import base64
import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import quote_plus

UnityPy = None
try:
    import UnityPy  # type: ignore
except Exception:
    UnityPy = None


DEFAULT_CARD_DB = Path(r"C:\Program Files\Wizards of the Coast\MTGA\MTGA_Data\Downloads\Raw")


@dataclass
class CardRecord:
    name: str
    grp_id: int
    art_id: int


class CardAssetLoader:
    def __init__(
        self,
        card_db_path: Optional[Path] = None,
        asset_dir: Optional[Path] = None,
    ) -> None:
        self.card_db_path = card_db_path or self._find_card_db()
        self.asset_dir = asset_dir or (self.card_db_path.parent.parent / "AssetBundle" if self.card_db_path else None)
        self.name_to_card: Dict[str, CardRecord] = {}
        self.grp_to_card: Dict[int, CardRecord] = {}
        self.art_cache: Dict[int, Optional[str]] = {}
        if self.card_db_path and self.card_db_path.exists():
            self._load_cards()

    def _find_card_db(self) -> Optional[Path]:
        if DEFAULT_CARD_DB.exists():
            candidates = sorted(DEFAULT_CARD_DB.glob("Raw_CardDatabase_*.mtga"))
            if candidates:
                return candidates[-1]
        return None

    def _load_cards(self) -> None:
        con = sqlite3.connect(self.card_db_path)
        cur = con.cursor()
        rows = cur.execute(
            """
            SELECT c.GrpId, c.ArtId, l.Loc
            FROM Cards c
            JOIN Localizations_enUS l ON c.TitleId = l.LocId
            """
        ).fetchall()
        con.close()
        for grp_id, art_id, name in rows:
            if not name:
                continue
            key = name.strip().lower()
            rec = CardRecord(name=name.strip(), grp_id=int(grp_id), art_id=int(art_id))
            if key not in self.name_to_card:
                self.name_to_card[key] = rec
            self.grp_to_card[int(grp_id)] = rec

    def find_by_name(self, name: str) -> Optional[CardRecord]:
        return self.name_to_card.get(name.strip().lower())

    def art_uri_for_name(self, name: str) -> Optional[str]:
        rec = self.find_by_name(name)
        if not rec:
            return None
        return self.art_uri_for_art_id(rec.art_id)

    def art_uri_for_art_id(self, art_id: int) -> Optional[str]:
        if art_id in self.art_cache:
            return self.art_cache[art_id]
        uri = self._extract_art(art_id)
        self.art_cache[art_id] = uri
        return uri

    def scryfall_image_url(self, card_name: str, version: str = "art_crop") -> str:
        """
        Build a Scryfall image URL without making a request.
        Example: https://api.scryfall.com/cards/named?format=image&version=art_crop&exact=Card+Name
        """
        return f"https://api.scryfall.com/cards/named?format=image&version={quote_plus(version)}&exact={quote_plus(card_name)}"

    def _extract_art(self, art_id: int) -> Optional[str]:
        if UnityPy is None or self.asset_dir is None:
            return None
        bundle_candidates = list(self.asset_dir.glob(f"{art_id:06d}_CardArt_*.mtga"))
        if not bundle_candidates:
            return None
        bundle_path = bundle_candidates[0]
        try:
            env = UnityPy.load(bundle_path)
            for obj in env.objects:
                if obj.type.name != "Texture2D":
                    continue
                tex = obj.read()
                image = tex.image
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                return f"data:image/png;base64,{b64}"
        except Exception:
            return None
        return None


__all__ = ["CardAssetLoader", "CardRecord"]
