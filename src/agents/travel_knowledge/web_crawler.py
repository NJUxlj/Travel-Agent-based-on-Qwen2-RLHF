import requests  
from bs4 import BeautifulSoup  
import os  

# 定义要爬取的城市列表  
cities = ["北京", "上海", "广州", "成都"]  

# 创建一个保存文件的目录  
os.makedirs("tour_pages", exist_ok=True)  

# 基本的携程旅游页面URL格式（需要根据实际页面调整）  
base_url = "https://you.ctrip.com/travels/"  

for city in cities:  
    # 将城市名称转化为url参数（需进行编码）  
    city_encoded = requests.utils.quote(city)  
    url = f"{base_url}{city_encoded}"  

    print(f"正在爬取: {url}")  
    response = requests.get(url)  

    if response.status_code == 200:  
        soup = BeautifulSoup(response.text, 'html.parser')  
        
        
    
        # 解析正文内容（根据实际网页结构调整选择器）  
        content = soup.find('div', class_='content')  # 示例选择器  
        if content:  
            # 保存到对应的文件  
            file_path = os.path.join("tour_pages", f"{city}.txt")  
            with open(file_path, 'w', encoding='utf-8') as f:  
                f.write(content.get_text(strip=True))  
                print(f"已保存 {city} 的旅游信息到 {file_path}")  
        else:  
            print(f"未找到 {city} 的旅游内容")  
    else:  
        print(f"请求失败: {response.status_code} - {url}")  

print("爬取完成")  