import os
from pathlib import Path
from data_collector import DataCollector
from data_collector import ProjectData

def setup_data_collector():
    
    current_dir = Path.cwd()
    projects_folder = current_dir / 'DataCollection'
    projects_folder.mkdir(exist_ok=True)

    print(f"Created DataCollector projects folder at: {projects_folder}")

    
    return DataCollector(str(projects_folder))

def run_test(collector):
  
    test_input = '''
    {
      "id": "tester",
      "domain": "https://www.scrapethissite.com/pages/",
      "docsSource": "web",
      "queries": ["Boeing still stuck?", "Bring the astronauts home"]
    }
    '''
    # test_input = '''
    # {
    # "id": "smu_test",
    # "domain": "https://catalog.smu.edu/preview_entity.php?catoid=64&ent_oid=6783",
    # "docsSource": "web",
    # "queries": ["ipsom lorem", "lorem ipson"]
    # }
    # '''
    result = collector.process(test_input)
    print("\nDataCollector Result:")
    print(result)

    result_data = ProjectData.from_json(result)
    print(f"\nCollected data saved to: {result_data.textData}")

if __name__ == '__main__':
    collector = setup_data_collector()
    run_test(collector)

