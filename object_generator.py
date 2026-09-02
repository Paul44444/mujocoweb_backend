import json
import os
import urllib.error
import urllib.request


SHAPES = {"sphere", "box", "capsule", "cylinder", "ellipsoid"}
MAX_PARTS = 6

OBJECT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "summary", "parts"],
    "properties": {
        "name": {"type": "string", "maxLength": 48},
        "summary": {"type": "string", "maxLength": 180},
        "parts": {
            "type": "array", "minItems": 1, "maxItems": MAX_PARTS,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["shape", "size", "position", "euler", "rgba", "mass"],
                "properties": {
                    "shape": {"type": "string", "enum": sorted(SHAPES)},
                    "size": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
                    "position": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
                    "euler": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
                    "rgba": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "number"}},
                    "mass": {"type": "number"},
                },
            },
        },
    },
}


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


def validate_object_spec(value):
    if not isinstance(value, dict) or not isinstance(value.get("parts"), list):
        raise ValueError("Invalid generated object")
    if not 1 <= len(value["parts"]) <= MAX_PARTS:
        raise ValueError(f"Objects require between 1 and {MAX_PARTS} parts")

    clean = {
        "name": str(value.get("name", "Generated object"))[:48],
        "summary": str(value.get("summary", "Custom generated object"))[:180],
        "parts": [],
    }
    for part in value["parts"]:
        shape = str(part.get("shape", "box")).lower()
        if shape not in SHAPES:
            raise ValueError(f"Unsupported shape: {shape}")
        size = part.get("size", [])
        position = part.get("position", [])
        euler = part.get("euler", [])
        rgba = part.get("rgba", [])
        if not all(isinstance(v, (int, float)) for values in (size, position, euler, rgba) for v in values):
            raise ValueError("Object parameters must be numeric")
        if (len(size), len(position), len(euler), len(rgba)) != (3, 3, 3, 4):
            raise ValueError("Invalid object parameter dimensions")
        clean["parts"].append({
            "shape": shape,
            "size": [_clamp(v, 0.005, 0.09) for v in size],
            "position": [_clamp(v, -0.12, 0.12) for v in position],
            "euler": [_clamp(v, -3.142, 3.142) for v in euler],
            "rgba": [_clamp(v, 0.0, 1.0) for v in rgba],
            "mass": _clamp(part.get("mass", 0.1), 0.005, 1.5),
        })
    return clean


def _fallback(description):
    text = description.lower()
    color = [0.22, 0.62, 0.95, 1.0]
    for word, rgba in {
        "red": [0.85, 0.18, 0.16, 1], "rot": [0.85, 0.18, 0.16, 1],
        "green": [0.25, 0.75, 0.25, 1], "grün": [0.25, 0.75, 0.25, 1],
        "yellow": [0.95, 0.72, 0.12, 1], "gelb": [0.95, 0.72, 0.12, 1],
        "purple": [0.62, 0.3, 0.85, 1], "lila": [0.62, 0.3, 0.85, 1],
    }.items():
        if word in text: color = rgba
    shape = "sphere" if any(w in text for w in ("ball", "kugel", "sphere")) else "cylinder" if any(w in text for w in ("bottle", "flasche", "zylinder")) else "box"
    size = [0.035, 0.035, 0.035]
    if any(w in text for w in ("small", "klein")): size = [0.022] * 3
    if any(w in text for w in ("large", "groß", "big")): size = [0.055] * 3
    return validate_object_spec({"name": "Local concept", "summary": description[:180], "parts": [{"shape": shape, "size": size, "position": [0, 0, 0], "euler": [0, 0, 0], "rgba": color, "mass": 0.18}]})


def generate_object(description):
    description = description.strip()
    if not 3 <= len(description) <= 600:
        raise ValueError("Describe the object in 3 to 600 characters")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        result = _fallback(description)
        result["generator"] = "local-fallback"
        return result

    payload = {
        "model": os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        "store": False,
        "input": [
            {"role": "developer", "content": "Design a creative, graspable MuJoCo object from the user's description. Compose it from 1-6 primitives. Dimensions and positions are metres, Euler angles radians, RGBA values 0-1, and masses kilograms. Keep the total object within roughly 18 cm and suitable for an Adroit hand. Use multiple parts when the concept benefits from them."},
            {"role": "user", "content": description},
        ],
        "text": {"format": {"type": "json_schema", "name": "mujoco_object", "strict": True, "schema": OBJECT_SCHEMA}},
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Object generation failed ({exc.code}): {detail}") from exc

    output_text = result.get("output_text")
    if not output_text:
        for item in result.get("output", []):
            for content in item.get("content", []):
                if content.get("type") == "output_text": output_text = content.get("text")
    if not output_text:
        raise RuntimeError("The model returned no object design")
    clean = validate_object_spec(json.loads(output_text))
    clean["generator"] = "ai"
    return clean
