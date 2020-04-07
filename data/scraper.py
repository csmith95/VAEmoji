import scrapy


VENDORS = ['apple', 'google', 'facebook', 'samsung', 'whatsapp', 'twitter']

class EmojiSpider(scrapy.Spider):
    name = "emoji_spider"
    start_urls = ['https://emojipedia.org/people/', 'https://emojipedia.org/nature/']

    def parse(self, response):
        LIST_SELECTOR = '.emoji-list'
        PATH_SELECTOR = 'a::attr(href)'

        for emoji_path in response.css(LIST_SELECTOR).css(PATH_SELECTOR).getall():

        	yield scrapy.Request(
                response.urljoin(emoji_path),
                callback=self.parseEmojiPage
            )

    def parseEmojiPage(self, response):

    	for vendor_container in response.css('.vendor-container'):
    		vendor = vendor_container.css('a').xpath('@href').extract_first().strip('/')
    		img_src = vendor_container.css('img').xpath('@src').extract_first()

    		if '.svg' in img_src or vendor not in VENDORS: continue

    		yield {
    		 	'image_url': img_src,
    		 	'vendor': vendor
    		}