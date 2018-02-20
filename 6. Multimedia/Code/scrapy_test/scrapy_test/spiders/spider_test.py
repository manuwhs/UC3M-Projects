
from scrapy.spider import BaseSpider
import os

# Basic Spider that downloads the htmls given and writes its content in files
# With the name of the folder.
class test_Spider(BaseSpider):
    name = "dmoz"     # Name of the spyder
    allowed_domains = ["dmoz.org"]  # Allowed domains
    start_urls = [      # Starting urls
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Books/",
        "http://www.dmoz.org/Computers/Programming/Languages/Python/Resources/"
    ]

    def parse(self, response):
        filename = response.url.split("/")[-2]
        with open(filename, 'wb') as f:
            f.write(response.body)

# Crawling spider that searches the web accordint to certain rules:

from scrapy.contrib.spiders import CrawlSpider, Rule
from scrapy.contrib.linkextractors.sgml import SgmlLinkExtractor

class IsBullshitSpider(CrawlSpider):
    name = 'isbullshit'
    start_urls = ['http://isbullsh.it'] # urls from which the spider will start crawling
    rules = [Rule(SgmlLinkExtractor(allow=[r'page/\d+']), follow=True), 
    	# r'page/\d+' : regular expression for http://isbullsh.it/page/X URLs
    	Rule(SgmlLinkExtractor(allow=[r'\d{4}/\d{2}/\w+']), callback='parse_doc')]
    	# r'\d{4}/\d{2}/\w+' : regular expression for http://isbullsh.it/YYYY/MM/title UR
         
      # The documents that fullfill the Rules will be downloaded and processed by 
     
    def parse_doc(self, response):
        filename = response.url.split("/")[-2]  # Writes a file with the name of the document 
        with open(filename, 'wb') as f:
            f.write(response.body)
            

class BESTUC3M_Crawler(CrawlSpider):
    name = 'BESTUC3M'
    allowed_domains = ["best.uc3m.es"]  # Allowed domains
    start_urls = ['http://best.uc3m.es/'] # urls from which the spider will start crawling
    rules = [Rule(SgmlLinkExtractor(allow=(r'', )), # Stracts all pages in the url, it follows the links in the domain and process everything
         callback='parse_doc', follow=True),
         ]
    def parse_doc(self, response):
        filename = response.url # Writes a file with the name of the document 
        # The first 3 positions are: 'http:', '', 'best.uc3m.es' Then the folder structure
        
        filename = filename[7:] # Eliminate the http:// part
        if (filename[-1] == '/'):
            filename = filename + "index"  # Give a name to the main file
            
        if not os.path.exists(os.path.dirname(filename)): # If the folder of this does not exist
            os.makedirs(os.path.dirname(filename))        # Make the folder
        with open(filename, 'wb') as f:
            f.write(response.body)
    # PROBLEM !!! When getting a resource, we just typt the Folder, and then the Server outputs the index.html  
            # but without that name. 
            
class cooking_Crawler(CrawlSpider):
    name = 'cooking'
    allowed_domains = ["www.homecookingadventure.com"]  # Allowed domains
    start_urls = ['http://www.homecookingadventure.com/'] # urls from which the spider will start crawling
    rules = [Rule(SgmlLinkExtractor(allow=(r'', )), # Stracts all pages in the url, it follows the links in the domain and process everything
         callback='parse_doc', follow=True),
         ]
    def parse_doc(self, response):
        filename = response.url # Writes a file with the name of the document 
        # The first 3 positions are: 'http:', '', 'best.uc3m.es' Then the folder structure
        
        filename = filename[7:] # Eliminate the http:// part

            
        if not os.path.exists(os.path.dirname(filename)): # If the folder of this does not exist
            os.makedirs(os.path.dirname(filename))        # Make the folder
        with open(filename, 'wb') as f:
            f.write(response.body)
    # PROBLEM !!! When getting a resource, we just typt the Folder, and then the Server outputs the index.html  
            # but without that name.    


class food_Crawler(CrawlSpider):
    name = 'food'
    allowed_domains = ["www.food.com"]  # Allowed domains
    start_urls = ['http://www.food.com/',
                  'http://www.food.com/topic/crock-pot-slow-cooker'] # urls from which the spider will start crawling
    rules = [Rule(SgmlLinkExtractor(allow=[r'/recipe/*']), # Stracts all pages in the folder recepi
         callback='parse_doc', follow=True),
         ]
    rules = [Rule(SgmlLinkExtractor(allow=[r'']), # Stracts all pages in the folder recepi
          follow=True),  # To follow more shit
         ]  
    def parse_doc(self, response):
        filename = response.url # Writes a file with the name of the document 
        # The first 3 positions are: 'http:', '', 'best.uc3m.es' Then the folder structure
        
        filename = filename[7:] # Eliminate the http:// part
        if (filename[-1] == '/'):
            filename = filename + "index"  # Give a name to the main file
            
        if not os.path.exists(os.path.dirname(filename)): # If the folder of this does not exist
            os.makedirs(os.path.dirname(filename))        # Make the folder
        with open(filename + ".html", 'wb') as f:
            f.write(response.body)
    # PROBLEM !!! When getting a resource, we just typt the Folder, and then the Server outputs the index.html  
            # but without that name.    
    
class all_recipes_spider(CrawlSpider):
    name = 'all_recipes'
    allowed_domains = ["allrecipes.com"]  # Allowed domains
    start_urls = ['http://allrecipes.com'] # urls from which the spider will start crawling
    rules = [Rule(SgmlLinkExtractor(allow=[r'.*(recipe).*']), # Stracts all pages in the folder recepi
         callback='parse_doc', follow=True),
         ]
    #.* means any number of chars
    rules = [Rule(SgmlLinkExtractor(allow=[r'']), # Stracts all pages in the folder recepi
          follow=True),  # To follow more shit
         ]  
    def parse_doc(self, response):
        filename = response.url # Writes a file with the name of the document 
        filename = filename[7:] # Eliminate the http:// part
        if (filename[-1] == '/'):   # If the end of the URL is just a folder name
            filename = filename + "index"  # Give a name to the main file
        if not os.path.exists(os.path.dirname(filename)): # If the folder of this does not exist
            os.makedirs(os.path.dirname(filename))        # Make the folder
        with open(filename + ".html", 'wb') as f:
            f.write(response.body)
        
    # PROBLEM !!! When getting a resource, we just typt the Folder, and then the Server outputs t
    # the index.html  but without that name ! We have to add some names.