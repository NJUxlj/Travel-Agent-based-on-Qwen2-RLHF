import os
import yaml
from pathlib import Path
import sys


from dataclasses import dataclass, field, asdict



@dataclass
class KGConfig:
    url: str = field(default="http://localhost")
    port: int = field(default=19530)
    username: str = field(default="neo4j")
    password: str = field(default="123456")