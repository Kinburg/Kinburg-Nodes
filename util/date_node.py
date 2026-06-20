from datetime import datetime

DATE_PRESETS = {
    "YYYY-MM-DD": "%Y-%m-%d",
    "YYYY_MM_DD": "%Y_%m_%d",
    "YYYY/MM/DD": "%Y/%m/%d",   # nested folders
    "DD-MM-YYYY": "%d-%m-%Y",
    "YYYY-MM": "%Y-%m",
    "YYYYMMDD": "%Y%m%d",
    "none": "",
    "custom": None,
}
TIME_PRESETS = {
    "HH-MM-SS": "%H-%M-%S",
    "HH-MM": "%H-%M",
    "HH": "%H",
    "HH:MM:SS": "%H:%M:%S",      # note: ':' is NOT valid in Windows paths/filenames
    "custom": None,
}


def _stamp(date_format, custom_date_format, include_time, time_format, custom_time_format):
    """Return the list of [date?, time?] strings for the current moment."""
    now = datetime.now()
    parts = []
    df = custom_date_format if date_format == "custom" else DATE_PRESETS.get(date_format, "")
    if df:
        parts.append(now.strftime(df))
    if include_time:
        tf = custom_time_format if time_format == "custom" else TIME_PRESETS.get(time_format, "")
        if tf:
            parts.append(now.strftime(tf))
    return parts


def _build(text, separator, date_format, custom_date_format, include_time, time_format, custom_time_format):
    chunks = ([text] if text != "" else []) + _stamp(
        date_format, custom_date_format, include_time, time_format, custom_time_format)
    return separator.join(chunks)


class KinburgDateString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": False, "tooltip": "Base text/path; the date (and time) is appended via 'separator'. Can be empty."}),
                "date_format": (list(DATE_PRESETS.keys()), {"default": "YYYY-MM-DD"}),
                "custom_date_format": ("STRING", {"default": "%Y-%m-%d", "tooltip": "strftime pattern, used when date_format = custom"}),
                "include_time": ("BOOLEAN", {"default": False}),
                "time_format": (list(TIME_PRESETS.keys()), {"default": "HH-MM"}),
                "custom_time_format": ("STRING", {"default": "%H-%M-%S", "tooltip": "strftime pattern, used when time_format = custom"}),
                "separator": ("STRING", {"default": "/", "tooltip": "Joins text / date / time. '/' creates subfolders in a save path; use '_' or '-' for a flat name."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "Kinburg-Nodes/util"

    def run(self, text, date_format, custom_date_format, include_time, time_format, custom_time_format, separator):
        return (_build(text, separator, date_format, custom_date_format,
                       include_time, time_format, custom_time_format),)

    @classmethod
    def IS_CHANGED(cls, text, date_format, custom_date_format, include_time, time_format, custom_time_format, separator):
        return _build(text, separator, date_format, custom_date_format,
                      include_time, time_format, custom_time_format)


NODE_CLASS_MAPPINGS = {"KinburgDateString": KinburgDateString}
NODE_DISPLAY_NAME_MAPPINGS = {"KinburgDateString": "Date String"}
