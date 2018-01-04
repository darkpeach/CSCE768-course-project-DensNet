import matplotlib.pyplot as plt

numbers = []
loss = []
accuracy = []

temp_loss = []
x = 0
with open('densenet.csv') as f:
	for cnt, line in enumerate(f):
		segments = line.rstrip('\n').split(',')
		temp_loss.append(float(segments[2]))
		if (cnt+1) % 100 == 0:
			loss.append(sum(temp_loss) / float(len(temp_loss)))
			temp_loss = []
			numbers.append(cnt)

plt.plot(numbers, loss)
plt.savefig('accuracy.png')
plt.ylabel('Accuracy')
plt.show()
