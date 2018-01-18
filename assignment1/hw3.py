maxS = 13


def ans(s):
	total = 0.0
	gamma = 0.8
	for i in range(s):
		if i is s-1:
			muliplier = 6.0
		else:
			muliplier = 0.0
		add = (gamma ** i) * (muliplier)
		total += add
	return total


for i in range(maxS):
	print ans(i)