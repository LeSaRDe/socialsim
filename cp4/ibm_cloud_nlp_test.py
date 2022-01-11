import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


if __name__ == '__main__':

    authenticator = IAMAuthenticator('QvFm1awXyWGDyI74jrdvdlRIy8_v75-efHMW6hnwGM9X')
    service = NaturalLanguageUnderstandingV1(
        version='2019-07-12',
        authenticator=authenticator)
    service.set_service_url('https://api.us-south.natural-language-understanding.watson.cloud.ibm.com')

    response = service.analyze(
        text='Bruce Banner is the Hulk and Bruce Wayne is BATMAN! '
             'Superman fears not Banner, but Wayne.',
        features=Features(entities=EntitiesOptions(),
                          keywords=KeywordsOptions(),
                          sentiment=SentimentOptions(targets=['superman']))).get_result()

    print(json.dumps(response, indent=2))