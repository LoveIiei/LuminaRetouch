"""
LuminaRetouch - Templates System
Pre-defined retouching presets for different photography styles.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class RetouchTemplate:
    """A retouching template with pre-defined settings."""
    name: str
    category: str
    description: str

    # Face Enhancement
    face_enhancement: float = 0.0

    # Skin
    skin_smoothness: float = 0.0
    blemish_removal: float = 0.0

    # Eyes
    eye_brightness: float = 0.0
    eye_size: float = 0.0
    dark_circle_removal: float = 0.0

    # Mouth
    teeth_whitening: float = 0.0
    lip_saturation: float = 0.0
    smile_enhancement: float = 0.0

    # Face Shape
    face_slimming: float = 0.0
    nose_slimming: float = 0.0
    chin_adjustment: float = 0.0
    jawline_sharpen: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'RetouchTemplate':
        return cls(**data)


# Built-in templates
BUILTIN_TEMPLATES: List[RetouchTemplate] = [
    # ===== Natural / Subtle =====
    RetouchTemplate(
        name="Natural Touch",
        category="Natural",
        description="Subtle enhancements that look unedited",
        skin_smoothness=0.15,
        blemish_removal=0.3,
        eye_brightness=0.1,
        teeth_whitening=0.1,
    ),

    RetouchTemplate(
        name="Fresh & Clean",
        category="Natural",
        description="Clean, fresh look with minimal retouching",
        skin_smoothness=0.2,
        blemish_removal=0.4,
        eye_brightness=0.15,
        dark_circle_removal=0.2,
        teeth_whitening=0.15,
    ),

    # ===== Portrait =====
    RetouchTemplate(
        name="Portrait Standard",
        category="Portrait",
        description="Professional portrait retouching",
        face_enhancement=0.2,
        skin_smoothness=0.3,
        blemish_removal=0.5,
        eye_brightness=0.2,
        dark_circle_removal=0.3,
        teeth_whitening=0.2,
        face_slimming=0.1,
    ),

    RetouchTemplate(
        name="Portrait Pro",
        category="Portrait",
        description="Enhanced portrait with face shaping",
        face_enhancement=0.25,
        skin_smoothness=0.35,
        blemish_removal=0.6,
        eye_brightness=0.25,
        eye_size=0.1,
        dark_circle_removal=0.4,
        teeth_whitening=0.25,
        face_slimming=0.15,
        nose_slimming=0.1,
        jawline_sharpen=0.15,
    ),

    # ===== Beauty / Glamour =====
    RetouchTemplate(
        name="Beauty Soft",
        category="Beauty",
        description="Soft, glamorous beauty look",
        face_enhancement=0.3,
        skin_smoothness=0.45,
        blemish_removal=0.7,
        eye_brightness=0.3,
        eye_size=0.15,
        dark_circle_removal=0.5,
        teeth_whitening=0.3,
        lip_saturation=0.2,
        face_slimming=0.15,
        nose_slimming=0.1,
    ),

    RetouchTemplate(
        name="Beauty Glam",
        category="Beauty",
        description="Full glamour retouching",
        face_enhancement=0.35,
        skin_smoothness=0.5,
        blemish_removal=0.8,
        eye_brightness=0.35,
        eye_size=0.2,
        dark_circle_removal=0.6,
        teeth_whitening=0.35,
        lip_saturation=0.3,
        smile_enhancement=0.1,
        face_slimming=0.2,
        nose_slimming=0.15,
        chin_adjustment=0.1,
        jawline_sharpen=0.2,
    ),

    # ===== Wedding =====
    RetouchTemplate(
        name="Wedding Classic",
        category="Wedding",
        description="Timeless wedding portrait look",
        face_enhancement=0.2,
        skin_smoothness=0.35,
        blemish_removal=0.6,
        eye_brightness=0.2,
        dark_circle_removal=0.4,
        teeth_whitening=0.3,
        smile_enhancement=0.1,
        face_slimming=0.1,
    ),

    RetouchTemplate(
        name="Wedding Elegant",
        category="Wedding",
        description="Elegant bridal retouching",
        face_enhancement=0.25,
        skin_smoothness=0.4,
        blemish_removal=0.7,
        eye_brightness=0.25,
        eye_size=0.1,
        dark_circle_removal=0.5,
        teeth_whitening=0.35,
        lip_saturation=0.15,
        smile_enhancement=0.15,
        face_slimming=0.15,
        nose_slimming=0.1,
        jawline_sharpen=0.1,
    ),

    # ===== ID Photo =====
    RetouchTemplate(
        name="ID Photo Standard",
        category="ID Photo",
        description="Clean look for official documents",
        skin_smoothness=0.2,
        blemish_removal=0.4,
        eye_brightness=0.1,
        dark_circle_removal=0.3,
    ),

    RetouchTemplate(
        name="ID Photo Enhanced",
        category="ID Photo",
        description="Polished ID photo look",
        face_enhancement=0.15,
        skin_smoothness=0.25,
        blemish_removal=0.5,
        eye_brightness=0.15,
        dark_circle_removal=0.4,
        teeth_whitening=0.1,
        face_slimming=0.05,
    ),

    # ===== Children =====
    RetouchTemplate(
        name="Kids Natural",
        category="Children",
        description="Light touch for children's photos",
        skin_smoothness=0.1,
        blemish_removal=0.3,
        eye_brightness=0.15,
    ),

    # ===== Business =====
    RetouchTemplate(
        name="Corporate Headshot",
        category="Business",
        description="Professional business portrait",
        face_enhancement=0.2,
        skin_smoothness=0.25,
        blemish_removal=0.5,
        eye_brightness=0.15,
        dark_circle_removal=0.4,
        teeth_whitening=0.15,
        jawline_sharpen=0.1,
    ),

    # ===== Fashion =====
    RetouchTemplate(
        name="Fashion Editorial",
        category="Fashion",
        description="High-fashion editorial look",
        face_enhancement=0.35,
        skin_smoothness=0.5,
        blemish_removal=0.9,
        eye_brightness=0.3,
        eye_size=0.15,
        dark_circle_removal=0.6,
        lip_saturation=0.25,
        face_slimming=0.2,
        nose_slimming=0.15,
        jawline_sharpen=0.25,
    ),

    # ===== Social Media =====
    RetouchTemplate(
        name="Instagram Ready",
        category="Social Media",
        description="Perfect for social media posts",
        face_enhancement=0.25,
        skin_smoothness=0.35,
        blemish_removal=0.6,
        eye_brightness=0.25,
        eye_size=0.1,
        dark_circle_removal=0.4,
        teeth_whitening=0.25,
        lip_saturation=0.2,
        face_slimming=0.15,
    ),

    RetouchTemplate(
        name="Selfie Perfect",
        category="Social Media",
        description="Enhanced selfie look",
        face_enhancement=0.3,
        skin_smoothness=0.4,
        blemish_removal=0.7,
        eye_brightness=0.3,
        eye_size=0.15,
        dark_circle_removal=0.5,
        teeth_whitening=0.3,
        smile_enhancement=0.1,
        face_slimming=0.2,
        nose_slimming=0.1,
    ),
]


class TemplateManager:
    """Manages retouching templates."""

    def __init__(self, custom_templates_path: Optional[Path] = None):
        self._builtin = {t.name: t for t in BUILTIN_TEMPLATES}
        self._custom: Dict[str, RetouchTemplate] = {}

        # Load custom templates
        self._custom_path = custom_templates_path or Path(__file__).parent / "custom_templates.json"
        self._load_custom_templates()

    def _load_custom_templates(self):
        """Load custom templates from disk."""
        if self._custom_path.exists():
            try:
                with open(self._custom_path, 'r') as f:
                    data = json.load(f)
                    for item in data:
                        template = RetouchTemplate.from_dict(item)
                        self._custom[template.name] = template
            except Exception as e:
                print(f"Error loading custom templates: {e}")

    def _save_custom_templates(self):
        """Save custom templates to disk."""
        try:
            data = [t.to_dict() for t in self._custom.values()]
            with open(self._custom_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving custom templates: {e}")

    def get_all_templates(self) -> List[RetouchTemplate]:
        """Get all templates (builtin + custom)."""
        return list(self._builtin.values()) + list(self._custom.values())

    def get_template(self, name: str) -> Optional[RetouchTemplate]:
        """Get a template by name."""
        return self._custom.get(name) or self._builtin.get(name)

    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        categories = set()
        for t in self.get_all_templates():
            categories.add(t.category)
        return sorted(categories)

    def get_templates_by_category(self, category: str) -> List[RetouchTemplate]:
        """Get all templates in a category."""
        return [t for t in self.get_all_templates() if t.category == category]

    def save_custom_template(self, template: RetouchTemplate):
        """Save a custom template."""
        template.category = "Custom" if not template.category else template.category
        self._custom[template.name] = template
        self._save_custom_templates()

    def delete_custom_template(self, name: str) -> bool:
        """Delete a custom template."""
        if name in self._custom:
            del self._custom[name]
            self._save_custom_templates()
            return True
        return False

    def is_custom(self, name: str) -> bool:
        """Check if a template is custom (editable)."""
        return name in self._custom
