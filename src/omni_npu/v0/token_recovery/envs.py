import os
from dataclasses import dataclass
@dataclass
class EnvVar:
    use_ha = os.environ.get("USE_HA", "0") == "1"
    ha_server_ip = os.environ.get("HA_SERVER_IP", os.environ.get("VLLM_DP_MASTER_IP"))
    ha_port: int = int(os.environ.get("HA_PORT", "4999"))

ENV = EnvVar()
print(ENV)