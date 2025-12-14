import os

READ_API_KEY = os.environ["THINGSPEAK_READ_API_KEY"]
WRITE_API_KEY = os.environ["THINGSPEAK_WRITE_API_KEY"]
CHANNEL_ID = os.environ["THINGSPEAK_CHANNEL_ID"]
WRITE_CHANNEL_ID = os.environ["THINGSPEAK_WRITE_CHANNEL_ID"]
USER_API_KEY = os.environ["THINGSPEAK_USER_API_KEY"]

MODEL_PATH = "models/svr_energy.pkl"

VOLTAGE = 'voltage'
CURRENT = 'current'
POWER = 'power'
ENERGY_KWH = 'energy_kwh'
TEMPERATURE = 'temperature'
HUMIDITY = 'humidity'
POWER_FACTOR = 'power_factor'
FREQUENCY = 'frequency'

FIELD_MAP = {
    'field1': 'voltage',
    'field2': 'current',
    'field3': 'power',
    'field4': 'energy_kwh',
    'field5': 'temperature',
    'field6': 'humidity',
    'field7': 'power_factor',
    'field8': 'frequency',
}

FIELDS = [VOLTAGE,
          CURRENT,
          POWER,
          ENERGY_KWH,
          TEMPERATURE,
          HUMIDITY,
          POWER_FACTOR,
          FREQUENCY
          ]


FEATURES = [VOLTAGE,
            CURRENT,
            POWER,
            TEMPERATURE,
            HUMIDITY,
            POWER_FACTOR,
            FREQUENCY
            ]

HOURS_AHEAD = 7 * 24
THINGSPEAK_RATE_SEC = 16  # safe margin

BASELINE_KWH = 5.0
