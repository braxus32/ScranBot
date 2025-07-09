import json
import re
import time
from contextlib import nullcontext

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver import ActionChains, Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.expected_conditions import element_to_be_clickable, visibility_of
from selenium.webdriver.support.wait import WebDriverWait
import requests

def handle(cls):
    cls.handle_menu(cls)
    cls.handle_overlay(cls)
    return cls

@handle
class ScranGetter:

    # Initialize selenium and open webpage
    driver = webdriver.Firefox()
    driver.get("https://scrandle.com/")
    wait = WebDriverWait(driver, timeout=10)

    def __init__(self):
        pass

    def handle_new_game(self):
        self.handle_overlay()
        self.handle_menu()

    def handle_menu(self):
        start_practice_button = self.driver.find_element(By.ID, "startPracticeButton")
        start_practice_button.click()

    def handle_overlay(self):
        time.sleep(1)
        ActionChains(self.driver) \
            .send_keys(Keys.ESCAPE) \
            .perform()
        time.sleep(1)

    def record(self):
        # Get scran data from content currently displayed
        html = self.driver.page_source
        soup = BeautifulSoup(html, features="html.parser")
        data_pair = []
        for scran in soup.find(id='scranArea').children:

            # Get and store the scran image
            img_url = re.search("scran_img.*webp", scran["style"])[0]
            local_img_url = "./images/" + img_url.split("/")[1]

            with open(local_img_url, 'wb') as writer:
                response = requests.get('https://scrandle.com/' + img_url, stream=True)

                if not response.ok:
                    print(response)

                for block in response.iter_content(1024):
                    if not block:
                        break

                    writer.write(block)

            # Get and store the scran facts
            facts = []
            for string in scran.stripped_strings:
                facts.append(string)
            host_date = facts[0].split(", ")
            host = host_date[0]
            year = host_date[1]
            country = facts[1]
            title = facts[2]
            subtext = facts[3].split(" • ")

            if len(subtext) == 1:
                desc = ""
                price = subtext[0].split("£")[1]
            else:
                desc = subtext[0]
                price = subtext[1].split("£")[1]

            data_pair.append({
                "image": local_img_url,
                "host": host,
                "year": year,
                "country": country,
                "title": title,
                "desc": desc,
                "price": price
            })

        return data_pair

    def play_binary(self, choice=False):
        data_pair = self.play(choice)
        return 0 if data_pair[0] > data_pair[1] else 1

    def play(self, choice=False):
        if choice: # right scran
            scran_choice = self.driver.find_element(By.CSS_SELECTOR, "div.scran:nth-child(2)")
        else: # left scran
            scran_choice = self.driver.find_element(By.CSS_SELECTOR, "div.scran:nth-child(1)")
        self.wait.until(element_to_be_clickable(scran_choice))
        scran_choice.click()
        time.sleep(1)

        # Collect result
        # print(self.driver.page_source)
        html = self.driver.page_source
        soup = BeautifulSoup(html, features="html.parser")
        data_pair = []
        for scran in soup.find(id='scranArea').children:
            facts = []
            for string in scran.stripped_strings:
                facts.append(string)
            rating = facts[2]
            data_pair.append(rating)

        time.sleep(3)
        return data_pair

    def collect_data(self, runs=100):
        with open('data.json', 'w') as df, open('key.json', 'w') as kf:
            for i in range(runs):
                if i != 0 and i % 10 == 0:
                    self.handle_new_game()
                data_pair = self.record()
                json.dump(data_pair, df)
                print(data_pair)
                key_pair = self.play()
                json.dump(key_pair, kf)
                print(key_pair)


    def quit(self):
        self.driver.quit()




def main():
    getter = ScranGetter()
    getter.collect_data()
    getter.quit()



if __name__ == "__main__":
    main()



