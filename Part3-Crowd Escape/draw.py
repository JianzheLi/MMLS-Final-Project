import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建画布
fig, ax = plt.subplots(figsize=(6, 6))

# 房间尺寸
room_width = 20
room_height = 10
door_width = 1  # 门的高度（竖直方向）

# 添加房间（矩形）
room = patches.Rectangle((0, 0), room_width, room_height, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(room)



# 添加中间门（蓝色）
door = patches.Rectangle((room_width, room_height / 2 - door_width / 2),
                         0.2, door_width,
                         linewidth=2, edgecolor='blue', facecolor='blue')
ax.add_patch(door)

# 添加出口文字（门外）
ax.text(room_width + 0.3, room_height / 2, 'exit', fontsize=12, va='center', color='red')

# 设置图形属性
ax.set_xlim(-1, room_width + 2)
ax.set_ylim(-1, room_height + 1)
ax.set_aspect('equal')
ax.axis('off')

# 保存图像
plt.savefig("room_single_blue_door.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
