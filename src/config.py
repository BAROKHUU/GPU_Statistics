"""Variable list and configuration for the two regression tasks."""

NUMERIC_COLUMNS = [
    "Memory", "Memory_Bandwidth", "Memory_Bus", "Memory_Speed",
    "Core_Speed", "Boost_Clock", "Process", "Release_Year", "Max_Power",
    "Release_Price",
]
COMMON_INPUTS = [
    "Memory", "Memory_Bandwidth", "Memory_Bus", "Memory_Speed",
    "Core_Speed", "Boost_Clock", "Process", "Release_Year",
    "Manufacturer", "Notebook_GPU", "Integrated",
]
TASKS = {
    "Max_Power": {
        "label": "Maximum power (W)",
        "inputs": COMMON_INPUTS,
        "log_target": False,
    },
    "Release_Price": {
        "label": "Release price (USD)",
        "inputs": COMMON_INPUTS + ["Max_Power"],
        "log_target": True,
    },
}

