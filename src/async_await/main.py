import time

def brewCoffee():
	time.sleep(4)
	return "Coffee ready"

def toastBagel():
	time.sleep(3)
	return "Toast ready"

def main():
	start_time = time.time()

	result_coffee = brewCoffee()
	print(result_coffee)
	result_bagel = toastBagel()
	print(result_bagel)

	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f'\nFinished in {elapsed_time:.2f} seconds');

if __name__ == '__main__':
	main()
