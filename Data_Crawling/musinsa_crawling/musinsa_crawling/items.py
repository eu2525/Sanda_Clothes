import scrapy

class MusinsaItem(scrapy.Item):
    product_name = scrapy.Field() # 제품명
    product_color = scrapy.Field()
    product_brand = scrapy.Field() # 브랜드
    product_category = scrapy.Field() # 제품 카테고리
    product_gender = scrapy.Field() # 성별
    product_url = scrapy.Field() # 품번
    product_image = scrapy.Field() # Url