# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/topics/items.html

from scrapy.item import Item, Field

class ScrapyTestItem(Item):
    # define the fields for your item here like:
    # name = Field()
    title = Field()  # Fields written to provide
    author = Field()
    tag = Field()
    date = Field()
    link = Field()
    
    pass
