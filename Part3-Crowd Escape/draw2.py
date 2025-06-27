import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 创建画布
fig, ax = plt.subplots(figsize=(6, 6))

# 房间尺寸
room_width = 20
room_height = 10
door_height = 1  # 每个出口的高度
wall_thickness = 0.2

# 添加房间轮廓（矩形框）
room = patches.Rectangle((0, 0), room_width, room_height, linewidth=2, edgecolor='black', facecolor='none')
ax.add_patch(room)

# 添加右上门（蓝色矩形表示）
right_door_top = patches.Rectangle((room_width, room_height - door_height), wall_thickness, door_height,
                                   linewidth=2, edgecolor='blue', facecolor='blue')
ax.add_patch(right_door_top)

# 添加右下门（蓝色矩形表示）
right_door_bottom = patches.Rectangle((room_width, 0), wall_thickness, door_height,
                                      linewidth=2, edgecolor='blue', facecolor='blue')
ax.add_patch(right_door_bottom)

# 添加出口文字
ax.text(room_width + 0.5, room_height - door_height / 2, 'exit', fontsize=12, va='center', color='red')
ax.text(room_width + 0.5, door_height / 2, 'exit', fontsize=12, va='center', color='red')

# 设置图形属性
ax.set_xlim(-1, room_width + 3)
ax.set_ylim(-1, room_height + 1)
ax.set_aspect('equal')
ax.axis('off')

# 保存图像
plt.savefig("room_with_two_doors_right_wall.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.show()
