#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import os

def scrape_messier_data():
    url = "https://www.astropixels.com/messier/messiercat.html"
    
    print(f"正在爬取数据: {url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table')
        if not table:
            print("未找到数据表格")
            return None
            
        rows = table.find_all('tr')
        print(f"找到 {len(rows)} 行数据")
        
        messier_data = []
        
        for i, row in enumerate(rows):
            if i == 0:
                continue
                
            cells = row.find_all(['td', 'th'])
            if len(cells) < 8:
                continue
                
            try:
                messier_num = cells[0].get_text(strip=True)
                ngc_num = cells[1].get_text(strip=True)
                common_name = cells[2].get_text(strip=True)
                object_type = cells[3].get_text(strip=True)
                ra_text = cells[4].get_text(strip=True)
                dec_text = cells[5].get_text(strip=True)
                constellation = cells[6].get_text(strip=True)
                magnitude = cells[7].get_text(strip=True)
                
                ra_decimal = parse_ra(ra_text)
                
                dec_decimal = parse_dec(dec_text)
                
                try:
                    mag_float = float(magnitude)
                except:
                    mag_float = 10.0
                
                best_season = get_best_season(ra_decimal)
                
                messier_data.append({
                    'Messier': messier_num,
                    'NGC': ngc_num,
                    'Common_Name': common_name,
                    'Object_Type': object_type,
                    'RA_Hours': ra_text,
                    'RA_Decimal': ra_decimal,
                    'Dec_Text': dec_text,
                    'Dec_Decimal': dec_decimal,
                    'Constellation': constellation,
                    'Magnitude': mag_float,
                    'Best_Season': best_season
                })
                
            except Exception as e:
                print(f"解析第 {i} 行时出错: {e}")
                continue
        
        print(f"成功解析 {len(messier_data)} 个天体")
        return messier_data
        
    except Exception as e:
        print(f"爬取数据失败: {e}")
        return None

def parse_ra(ra_text):
    try:
        match = re.search(r'(\d+):(\d+)', ra_text)
        if match:
            hours = int(match.group(1))
            minutes = int(match.group(2))
            return (hours + minutes/60.0) * 15.0
        else:
            return 0.0
    except:
        return 0.0

def parse_dec(dec_text):
    try:
        match = re.search(r'([+-]?)(\d+):(\d+)', dec_text)
        if match:
            sign = -1 if match.group(1) == '-' else 1
            degrees = int(match.group(2))
            minutes = int(match.group(3))
            return sign * (degrees + minutes/60.0)
        else:
            return 0.0
    except:
        return 0.0

def get_best_season(ra_decimal):
    if 330 <= ra_decimal or ra_decimal < 60:
        return 'winter'
    elif 60 <= ra_decimal < 150:
        return 'spring'
    elif 150 <= ra_decimal < 240:
        return 'summer'
    else:
        return 'autumn'

def save_to_csv(messier_data, filename='messier_catalog.csv'):
    if not messier_data:
        print("没有数据可保存")
        return False
        
    df = pd.DataFrame(messier_data)
    
    filepath = os.path.join('..', filename)
    df.to_csv(filepath, index=False, encoding='utf-8')
    print(f"数据已保存到: {filepath}")
    print(f"共保存 {len(df)} 个天体记录")
    
    print("\n数据概览:")
    print(f"天体类型分布:")
    print(df['Object_Type'].value_counts())
    print(f"\n季节分布:")
    print(df['Best_Season'].value_counts())
    
    return True

def main():
    print("=== Messier天体数据爬虫 ===")
    
    messier_data = scrape_messier_data()
    
    if messier_data:
        save_to_csv(messier_data)
        print("\n✓ 爬虫任务完成")
    else:
        print("\n✗ 爬虫任务失败")

if __name__ == '__main__':
    main()
