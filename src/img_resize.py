from PIL import Image

animal = ['cheetah', 'leopard', 'tiger']

# img.size

for i in range(3213, 4278):
    img = Image.open(f'./data/{animal[2]}/{animal[2]}_{i}.jpg')
    # img = Image.open(f'./data/{animal[2]}/{animal[2]}_{i}.jpg')
    img = img.resize((128, 128))
    img.save(f'./data/{animal[2]}_resize/{animal[2]}_{i}.jpg')
    

