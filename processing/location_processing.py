import requests


def get_country_from_location_element(location_text):
    if location_text == '' or location_text is None:
        return 'Unknown'
    if 'Lives in' in location_text:
        location_text = location_text.replace('Lives in', '').strip()   
    if 'Lived in' in location_text:
        location_text = location_text.replace('Lived in', '').strip()

    if ',' in location_text:
        location_text = location_text.split(',')[-1] 
    return check_location(location_text.strip())


def check_location(location_text):
    country_request = requests.get(f'https://restcountries.com/v3.1/name/{location_text}?fullText=true')
    if country_request.ok:
        return location_text
    else:
        return 'America'
    
# print(get_country_from_location_element(''))