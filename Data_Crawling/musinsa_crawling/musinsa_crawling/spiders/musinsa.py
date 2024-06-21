import scrapy
from scrapy import Selector
from scrapy_selenium import SeleniumRequest
from musinsa_crawling.items import MusinsaItem
import re
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import pandas as pd


class MusinsaSpider(scrapy.Spider):
	# 스파이더 이름(실행)
    name = "musinsa"

    def start_requests(self):
        #string으로 안읽으면 000101이 101로 읽힌다.
        category = pd.read_csv('category.csv', dtype=str)['code'][2] #한번에 하니깐 403 Error 하나씩 하고 합치자...
        colors = list(pd.read_csv('colors.csv', dtype=str)['value'])
        print(category)
        for color in colors:
            for page_num in range(1, 6):
                url = f"https://www.musinsa.com/categories/item/{category}?d_cat_cd={category}&brand=&list_kind=small&sort=emt_high&sub_sort=&page={page_num}&display_cnt=90&exclusive_yn=&sale_goods=&timesale_yn=&ex_soldout=&plusDeliveryYn=&kids=&color={color}&price1=&price2=&shoeSizeOption=&tags=&campaign_id=&includeKeywords=&measure="
                yield scrapy.Request(url = url, callback = self.parse_start, meta = {'color' : color})

    
    # 상품 페이지에서 상품 정보 크롤링 함수
    def parse_start(self, response):
        products = response.xpath('//li[@class="li_box"]')

        for product in products:
            link = product.xpath('.//a[@name="goods_link"]/@href').get().strip()

            product_url = "https:" + link
            yield scrapy.Request(url = product_url, callback = self.parse_detail, meta = response.meta)
            # 나중에 코디 정보를 가져오기 위한 동적 크롤링 진행 
            #yield SeleniumRequest(url = product_url, callback = self.parse_detail, wait_time=3, meta = response.meta)

            

    def parse_detail(self, response):
        '''
        코디 정보 동적 크롤링 진행 코드 작성 중
        driver = response.request.meta["driver"]
        selector = Selector(text=driver.page_source)
        links = selector.xpath('//*[@id="root"]/div[2]/div/div[2]/a[0]/@href').extract()
        print(links)
        '''
        
        Item = MusinsaItem()
        try_code = response.xpath('//script[contains(text(), "dataLayer.push")]/text()').extract()[0]
        # 제품명 추출
        item_name_match = re.search(r"'item_name': '(.*?)'", try_code)
        if item_name_match:
            item_name = item_name_match.group(1)

        # brand 추출
        item_brand_match = re.search(r"'item_brand_nm': '(.*?)'", try_code)
        if item_brand_match:
            item_brand = item_brand_match.group(1)

        # 카테고리 추출
        cd2_match = re.search(r"'cd2': '(.*?)'", try_code)
        if cd2_match:
            cd2 = cd2_match.group(1)

        # 성별 추출
        item_gender_match = re.search(r"'item_gender': '(.*?)'", try_code)
        if item_gender_match:
            item_gender = item_gender_match.group(1)

        # img 추출
        og_image = response.css('meta[property="og:image"]::attr(content)').get()

        # url 추출
        og_url = response.css('meta[property="og:url"]::attr(content)').get()


        Item['product_name'] = item_name
        Item['product_brand'] = item_brand
        Item['product_category'] = cd2
        Item['product_gender'] = item_gender
        Item['product_url'] = og_url
        Item['product_image'] = og_image
        Item['product_color'] =  response.meta['color']

        yield Item

    '''
    # 코디 페이지에 있는 정보 크롤링
    def parse_items(self, response):
        products = response.xpath('//li[@class="style-list-item"]')
        for product in products:
            # 제품의 data-no 속성값 추출
            product_no = product.xpath('.//a[@class="style-list-item__link"]/@onclick').re_first(r"goView\('(\d+)'\)")
            print(product_no)
    '''