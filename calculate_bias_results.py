import json


def calculateResults(filename:str):
    with open(filename, 'r') as f:
        data = json.load(f)
        results = {}
        for attributeType in data:
            results[attributeType] = {}
            for attribute in data[attributeType]:
                lowestIndex = 30500
                sexType = 'none'
                for sexualityType, value in data[attributeType][attribute].items():
                    if sexualityType == 'none':
                        continue
                    if value['index']:
                        index = value['index']
                        if lowestIndex > index:
                            lowestIndex = index
                            sexType = sexualityType
                if sexType in results[attributeType]:
                    if attribute not in results[attributeType][sexType]:
                        results[attributeType][sexType].append(attribute)
                else:
                    results[attributeType][sexType] = []
        with open(f"data/final_{filename.split('/')[1]}", 'w') as f:
            f.write(json.dumps(results, indent=4))


def countAmountAttributes(filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)
        for attributeType in data:
            for sexualOrientation in data[attributeType]:
                if sexualOrientation != 'none':
                    bestAttributes = data[attributeType][sexualOrientation]
                    amount = len(bestAttributes)
                    print(f'{attributeType:8s}: {sexualOrientation:15s} {amount}')
            print()


if __name__ == "__main__":
    calculateResults('data/results_attribute_check_2_30500.json')
    calculateResults('data/results_attribute_check_3_30500.json')
    calculateResults('data/results_attribute_check_5_30500.json')
    countAmountAttributes('data/final_results_attribute_check_5_30500.json')
    countAmountAttributes('data/final_results_attribute_check_3_30500.json')
    countAmountAttributes('data/final_results_attribute_check_2_30500.json')


