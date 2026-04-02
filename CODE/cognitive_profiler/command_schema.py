COMMAND_SCHEMA = {
    "type": "array",
    "description": "[REDACTED_BY_SCRIPT]",
    "items": {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "GOTO_URL"},
                    "params": {
                        "type": "object",
                        "properties": {"url": {"type": "string", "format": "uri"}},
                        "required": ["url"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "FILL_INPUT"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["selector", "value"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "CLICK_ELEMENT"},
                    "params": {
                        "type": "object",
                        "properties": {"selector": {"type": "string"}},
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "CHECK_OPTION"},
                    "params": {
                        "type": "object",
                        "properties": {"selector": {"type": "string"}},
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "PRESS_KEY"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "key": {"type": "string"}
                        },
                        "required": ["selector", "key"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "TRY_CLICK_ELEMENT"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "timeout": {"type": "number"}
                        },
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "WAIT_FOR_SELECTOR"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "timeout": {"type": "number"},
                            "state": {
                                "type": "string", 
                                "enum": ["visible", "hidden", "attached", "detached"],
                                "description": "[REDACTED_BY_SCRIPT]'visible'."
                            }
                        },
                        "required": ["selector"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "WAIT_FOR_NAVIGATION"},
                    "params": {"type": "object"},
                    "description": {"type": "string"}   
                },
                "required": ["command", "params"]
            },
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "const": "SELECT_OPTION"},
                    "params": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "value": {"type": "string"}
                        },
                        "required": ["selector", "value"]
                    },
                    "description": {"type": "string"}
                },
                "required": ["command", "params"]
            }
        ]
    }
}