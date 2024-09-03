################################### Schema for camera Info ###################################
cameraInfo_schema = {
    "type": "object",
    "patternProperties": {
        "^.*$": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "tracker": {
                        "type": "object",
                        "properties": {
                            "model_id": {"type": "string"},
                            "roi": {
                                "type": "object",
                                "properties": {
                                    "direction": {"type": "string"},
                                    "line": {"type": "number"},
                                    "coordinates": {
                                                "type": "array",
                                                "items": {"type": "number"}
                                            },
                                    "type": {"type": "string"},
                                    "max_distance" : {"type": "number"}
                                },
                                "required": ["direction", "coordinates", "type", "line"],
                                "additionalProperties": True
                            }
                        },
                        "required": ["model_id", "roi"],
                        "additionalProperties": True
                    },
                    "cropping": {
                        "type": "object",
                        "patternProperties": {
                            "^.*$": {
                                "type": "object",
                                "properties": {
                                    "roi": {
                                        "type": "array",
                                        "items": {"type": "number"},
                                        "minItems": 4,
                                        "maxItems": 4
                                    },
                                    "model_1": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string"},
                                                "model_id": {"type": "string"},
                                                "threshold": {
                                                    "type": "object",
                                                    "additionalProperties": True
                                                }
                                            },
                                            "required": ["type", "model_id", "threshold"],
                                            "additionalProperties": True
                                        },
                                        "minItems": 1
                                    },
                                    "model_2": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "type": {"type": "string"},
                                                "model_id": {"type": "string"},
                                                "threshold": {
                                                    "type": "object",
                                                    "additionalProperties": True
                                                },
                                                "class_name": {"type": "string"}
                                            },
                                            "required": ["type", "model_id", "threshold", "class_name"],
                                            "additionalProperties": True
                                        },
                                        "minItems": 0
                                    }
                                },
                                "required": ["roi", "model_1" ],
                                "additionalProperties": True
                            }
                        }
                    }
                },
                "required": ["steps", "cropping"],
                "additionalProperties": True
            }
        }
    },
    "additionalProperties": True
}


################################### Schema for model Info ###################################
modelInfo_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string"
            },
            "type": {
                "type": "string"
            },
            "weights": {
                "type": "string"
            },
            "config": {
                "type": "string"
            },
            "modelKey": {
                "type": "integer",
                "enum": [0, 1, 2, 3]  # Allowed values for modelKey
            },
            "threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1  # Assuming threshold is between 0 and 1
            },
            "params": {
                "type": "object",
                # "properties": {
                #     "classes": {
                #         "type": "string"
                #         # "items": {
                #         #     "type": "string"
                #         # },
                #         # "minItems": 1  # At least one item should be there
                #     }
                },
            }
        },
        "required": ["id", "type", "weights", "config", "modelKey", "threshold", "params"],
        "additionalProperties": False
    }


################################### Schema for tracker Info ###################################
trackerInfo_schema = {
    "type": "array",
    "items": {
        "type": "string"
    }
}