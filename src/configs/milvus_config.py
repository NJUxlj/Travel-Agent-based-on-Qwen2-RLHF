import os
from typing import Dict, Any
import yaml
from pathlib import Path
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict



@dataclass
class MilvusConfig:
    url: str = field(default="http://localhost")
    port: int = field(default=19530)

    db_name: str = field(default="default")
    collection_name: str = field(default="default")