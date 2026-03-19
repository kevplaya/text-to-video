"""
Custom exception classes for text-to-video generator.
Each exception supports a `raw_text_preview` attribute for debugging.
"""


class GeminiNotConfigured(RuntimeError):
    pass


class GeminiEmptyResponse(RuntimeError):
    pass


class GeminiInvalidJSON(ValueError):
    def __init__(self, message: str, *, raw_text_preview: str = ""):
        super().__init__(message)
        self.raw_text_preview = raw_text_preview


class GeminiImageGenerationFailed(RuntimeError):
    def __init__(self, message: str, *, raw_text_preview: str = ""):
        super().__init__(message)
        self.raw_text_preview = raw_text_preview


class GeminiSpeechGenerationFailed(RuntimeError):
    def __init__(self, message: str, *, raw_text_preview: str = ""):
        super().__init__(message)
        self.raw_text_preview = raw_text_preview


class GrokNotConfigured(RuntimeError):
    pass


class GrokImageGenerationFailed(RuntimeError):
    def __init__(self, message: str, *, raw_text_preview: str = ""):
        super().__init__(message)
        self.raw_text_preview = raw_text_preview


class GrokVideoGenerationFailed(RuntimeError):
    pass


class VeoNotConfigured(RuntimeError):
    pass


class VeoVideoGenerationFailed(RuntimeError):
    pass


class KlingNotConfigured(RuntimeError):
    pass


class KlingVideoGenerationFailed(RuntimeError):
    pass


class GPTNotConfigured(RuntimeError):
    pass


class GPTEmptyResponse(RuntimeError):
    pass


class GPTInvalidJSON(ValueError):
    def __init__(self, message: str, *, raw_text_preview: str = ""):
        super().__init__(message)
        self.raw_text_preview = raw_text_preview


class ElevenLabsNotConfigured(RuntimeError):
    pass


class ElevenLabsSpeechGenerationFailed(RuntimeError):
    def __init__(self, message: str, *, raw_text_preview: str = ""):
        super().__init__(message)
        self.raw_text_preview = raw_text_preview
