import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax1 = plt.subplot(2, 2, 1)
ax1.set_title('ax1')
ax2 = plt.subplot(2, 2, 2)
ax2.set_title('ax2')
ax3 = plt.subplot(2, 1, 2)
ax3.set_title('ax3')
plt.tight_layout()
plt.savefig('test_plot.png')
