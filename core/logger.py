import json
import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


class Logger:
    _name = "oclaw"
    _configured = False
    _max_value_length = 300
    _sensitive_fields = {
        "message",
        "content",
        "result",
        "arguments",
        "prompt",
        "input",
        "output",
        "thinking",
    }
    _reset = "\033[0m"
    _level_colors = {
        logging.DEBUG: "\033[38;5;244m",
        logging.INFO: "\033[38;5;117m",
        logging.WARNING: "\033[38;5;220m",
        logging.ERROR: "\033[38;5;203m",
        logging.CRITICAL: "\033[38;5;197m",
    }
    _source_palette = (
        "\033[38;5;81m",
        "\033[38;5;112m",
        "\033[38;5;141m",
        "\033[38;5;214m",
        "\033[38;5;39m",
        "\033[38;5;177m",
        "\033[38;5;45m",
        "\033[38;5;149m",
    )

    def __init__(self, source: str):
        self.source = source
        self._logger = self._get_base_logger()

    @classmethod
    def get(cls, source: str) -> "Logger":
        cls._configure()
        return cls(source)

    @classmethod
    def _get_base_logger(cls) -> logging.Logger:
        cls._configure()
        return logging.getLogger(cls._name)

    @classmethod
    def _configure(cls) -> None:
        if cls._configured:
            return

        project_root = Path(__file__).resolve().parents[1]
        logs_dir = project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "oclaw.log"

        base_logger = logging.getLogger(cls._name)
        level_name = os.getenv("OCLAW_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        base_logger.setLevel(level)
        base_logger.propagate = False

        max_value_length_raw = os.getenv("OCLAW_LOG_MAX_VALUE_LENGTH")
        if max_value_length_raw and max_value_length_raw.isdigit():
            cls._max_value_length = int(max_value_length_raw)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(source)s | %(message)s",
            "%Y-%m-%dT%H:%M:%S",
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(cls._ConsoleFormatter())

        file_handler = TimedRotatingFileHandler(
            filename=log_file,
            when="midnight",
            interval=1,
            backupCount=7,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        base_logger.handlers.clear()
        base_logger.addHandler(stream_handler)
        base_logger.addHandler(file_handler)

        cls._configured = True

    class _ConsoleFormatter(logging.Formatter):
        def __init__(self) -> None:
            super().__init__(
                "%(asctime)s | %(levelname)s | %(source)s | %(message)s",
                "%Y-%m-%dT%H:%M:%S",
            )

        def format(self, record: logging.LogRecord) -> str:
            formatted = super().format(record)
            if not self._is_color_enabled():
                return formatted

            level_color = Logger._level_colors.get(record.levelno, "")
            source = str(getattr(record, "source", "unknown"))
            source_color = Logger._source_palette[
                sum(ord(char) for char in source) % len(Logger._source_palette)
            ]
            colored_level = f"{level_color}{record.levelname}{Logger._reset}"
            colored_source = f"{source_color}{source}{Logger._reset}"
            return formatted.replace(record.levelname, colored_level, 1).replace(
                source,
                colored_source,
                1,
            )

        @staticmethod
        def _is_color_enabled() -> bool:
            mode = os.getenv("OCLAW_LOG_COLORS", "auto").lower()
            if mode == "never":
                return False
            if mode == "always":
                return True
            return os.isatty(2)

    def info(self, event: str, **data: Any) -> None:
        self._log(logging.INFO, event, data)

    def debug(self, event: str, **data: Any) -> None:
        self._log(logging.DEBUG, event, data)

    def error(self, event: str, **data: Any) -> None:
        self._log(logging.ERROR, event, data)

    def warning(self, event: str, **data: Any) -> None:
        self._log(logging.WARNING, event, data)

    def _log(self, level: int, event: str, data: dict[str, Any]) -> None:
        payload = {"event": event, **self._compact_data(data)}
        message = json.dumps(payload, ensure_ascii=False, default=str)
        self._logger.log(level, message, extra={"source": self.source})

    @classmethod
    def _compact_data(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cls._compact_text(value)

        if isinstance(value, dict):
            compacted: dict[str, Any] = {}
            for key, val in value.items():
                normalized_key = str(key)
                if normalized_key.lower() in cls._sensitive_fields:
                    compacted[normalized_key] = cls._redacted_value(val)
                    continue
                compacted[normalized_key] = cls._compact_data(val)
            return compacted

        if isinstance(value, list):
            return [cls._compact_data(item) for item in value]

        if isinstance(value, tuple):
            return [cls._compact_data(item) for item in value]

        return value

    @classmethod
    def _redacted_value(cls, value: Any) -> dict[str, Any]:
        text = str(value)
        return {
            "redacted": True,
            "chars": len(text),
            "type": type(value).__name__,
        }

    @classmethod
    def _compact_text(cls, text: str) -> str:
        if len(text) <= cls._max_value_length:
            return text

        return (
            f"{text[: cls._max_value_length]}"
            f"... [truncated {len(text) - cls._max_value_length} chars]"
        )
