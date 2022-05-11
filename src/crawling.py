from selenium import webdriver 
from selenium.webdriver.common.keys import Keys 
import time 
import urllib.request
import base64

keyword = 'west highland white terrier'
breed = 'white_terrier'
dir_name = f'./data/{breed}'
chromedriver = '/Users/mingyu/dev/chromedriver' 

driver = webdriver.Chrome(chromedriver)
driver.implicitly_wait(3)

driver.get('https://www.google.co.kr/imghp?hl=ko') 
Keyword = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input') 
Keyword.send_keys(keyword) 
driver.find_element_by_xpath('//*[@id="sbtc"]/button').click()

elem = driver.find_element_by_tag_name("body") 
for i in range(15):
    print(i) 
    elem.send_keys(Keys.PAGE_DOWN) 
    time.sleep(0.1) 
    try: 
        driver.find_element_by_xpath('//*[@id="islmp"]/div/div/div/div[1]/div[4]/div[2]/input').click() 
        for _ in range(60): 
            elem.send_keys(Keys.PAGE_DOWN) 
            time.sleep(0.1) 
    except: pass

links = [] 
images = driver.find_elements_by_css_selector("img.rg_i.Q4LuWd") 
for image in images: 
    if image.get_attribute('src') != None: 
        links.append(image.get_attribute('src')) 

time.sleep(2)
# print(links)
# print(links[0])

download = 0
for i, url in enumerate(links):
    try:
        urllib.request.urlretrieve(url, f"./{dir_name}/{keyword}_{i}.jpg")
        download += 1
    except:
        try:
            img = base64.b64decode(url)
            urllib.request.urlretrieve(img, f"./{dir_name}/{keyword}_{i}.jpg")
            download += 1
        except: pass
        
    # print(f'{url} : download\n')
print(f'{keyword} 이미지 개수: {download}') 
driver.close()
