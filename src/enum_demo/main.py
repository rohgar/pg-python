from enum import Enum


class AssetLocationType(Enum):
    SHARD = "shard"


def main():
	asset_type = AssetLocationType("shard")
	print(asset_type)

	asset_type = AssetLocationType("foo")
	print(asset_type)

if __name__ == '__main__':
	main()
