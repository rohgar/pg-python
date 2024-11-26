import asyncio
import time

async def brewCoffee():
	await asyncio.sleep(4)
	return "Coffee ready"

async def toastBagel():
	await asyncio.sleep(3)
	return "Toast ready"

async def main():
	start_time = time.time()

	batch = asyncio.gather(brewCoffee(), toastBagel())
	result_coffee, result_bagel = await batch

	print(result_coffee)
	print(result_bagel)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f'\nFinished in {elapsed_time:.2f} seconds');

if __name__ == '__main__':
	asyncio.run(main())
