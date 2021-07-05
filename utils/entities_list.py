# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:34 PM

import json

#Entities_list = [
#    "company",
#    "address",
#    "date",
#    "total"
#]
config_file = open("config.json", "r")
config_file = json.load(config_file)

Entities_list = config_file["entities"]
