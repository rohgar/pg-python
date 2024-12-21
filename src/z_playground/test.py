#!/usr/bin/env python3

import json

sample_data = ['aasdasdasdasda', 'bbbmnbmhbkjhbmh']

print(f"sample_data = {sample_data}")

for row, rowvalue in enumerate(sample_data):
	if len(rowvalue) > 10:
		sample_data[row] = rowvalue[:3]
		
print(f"sample_data = {sample_data}")

verifierColumnClassification = None
if verifierColumnClassification:
	print('abc')
	

operational_cols_count = 19
total_columns_count = 19
percentage = round(operational_cols_count / total_columns_count * 100, 2)
print(f"percentage = {percentage:.2f}")


def get_owners(asset_xid: str) -> list[str]:
	try:
		raise Exception("FOO")
	except Exception as e:
		print(
			"Warning: Failed to get oncall list for '{classificationResponse.assetXid}' with error: {e}"
		)
	
result = get_owners("asd")
print(f"result = {result}")
print(f"result type = {type(result)}")

oncall_list = [[]]
if oncall_list:
	print("exiusts")
	
	
print("this is a %s", "foo")




from datetime import datetime
current_datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"current_datetime_str = {current_datetime_str}")