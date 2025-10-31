# Modified based on: https://github.com/HKUNLP/HumanPrompt/blob/main/humanprompt/tasks/dataset_loader.py
import os
from typing import Any, Dict, Union
import argparse
import datasets
from datasets import Dataset, DatasetDict, IterableDataset, IterableDatasetDict

DIR = os.path.join(os.path.dirname(__file__))


class DatasetDownLoader(object):
    """Dataset loader class."""

    # Datasets implemented in this repo
    # Either they do not exist in the datasets library or they have something improper
    own_dataset = {}

    @staticmethod
    def load_dataset(
        dataset_name: str,
        split: str = None,
        subset_name: str = None,
        dataset_key_map: Dict[str, str] = None,
        save_to_local = True,
        local_data_dir = None,
        cache_dir: str = None,
        **kwargs: Any
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        """
        Load dataset from the datasets library or from this repo.
        Args:
            dataset_name: name of the dataset
            split: split of the dataset
            subset_name: subset name of the dataset
            dataset_key_map: mapping original keys to a unified set of keys
            save_to_local: whether to save the dataset to local disk
            local_data_dir: directory to save the dataset locally
            cache_dir: directory to store the cached dataset
            **kwargs: arguments to pass to the dataset

        Returns: dataset

        """
        # Create cache directory if it doesn't exist
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        if local_data_dir and not os.path.exists(local_data_dir):
            os.makedirs(local_data_dir, exist_ok=True)
            
        # Check whether the dataset is in the own_dataset dictionary
        if dataset_name in DatasetDownLoader.own_dataset.keys():
            dataset_path = DatasetDownLoader.own_dataset[dataset_name]
            dataset = datasets.load_dataset(
                path=dataset_path,
                split=split,
                name=subset_name,
                cache_dir=cache_dir,
                **kwargs
            )
        else:
            # If not, load it from the datasets library
            dataset = datasets.load_dataset(
                path=dataset_name,
                split=split,
                name=subset_name,
                cache_dir=cache_dir,
                **kwargs
            )

        if dataset_key_map:
            reverse_dataset_key_map = {v: k for k, v in dataset_key_map.items()}
            dataset = dataset.rename_columns(reverse_dataset_key_map)
        
        # Save dataset to local disk if save_to_local is True and local_data_dir is provided
        if save_to_local and local_data_dir:
            # Create a unique save path based on dataset name and split
            save_path = os.path.join(local_data_dir, dataset_name.replace('/', '_'))
            if subset_name:
                save_path = f"{save_path}_{subset_name}"
            if split:
                save_path = f"{save_path}_{split}"
            
            print(f"Saving dataset to {save_path}")
            dataset.save_to_disk(save_path)

        return dataset



def parse_args():
    parser = argparse.ArgumentParser(description='Download and save Hugging Face datasets to local disk')

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help='Name of the dataset to download'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default=None,
        help='Split of the dataset (e.g., train, validation, test)'
    )
    
    parser.add_argument(
        '--subset_name',
        type=str,
        default=None,
        help='Subset name of the dataset'
    )
    
    parser.add_argument(
        '--dataset_key_map',
        type=str,
        default=None,
        help='JSON string mapping original keys to a unified set of keys'
    )
    
    parser.add_argument(
        '--save_to_local',
        action='store_true',
        default=True,
        help='Whether to save the dataset to local disk (default: True)'
    )
    
    parser.add_argument(
        '--no_save_to_local',
        action='store_true',
        help='Do not save the dataset to local disk'
    )
    
    parser.add_argument(
        '--local_data_dir',
        type=str,
        default=None,
        help='Directory to save the dataset locally'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=None,
        help='Directory to store the cached dataset'
    )
    
    parser.add_argument(
        '--kwargs',
        type=str,
        default=None,
        help='Additional keyword arguments as key=value pairs separated by commas'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Process dataset_key_map if provided
    dataset_key_map = None
    if args.dataset_key_map:
        import json
        try:
            dataset_key_map = json.loads(args.dataset_key_map)
            if not isinstance(dataset_key_map, dict):
                raise ValueError("dataset_key_map must be a JSON object/dictionary")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON string for dataset_key_map: {args.dataset_key_map}")
    
    # Process additional kwargs if provided
    kwargs = {}
    if args.kwargs:
        for kv_pair in args.kwargs.split(','):
            if '=' in kv_pair:
                key, value = kv_pair.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Try to convert value to appropriate type
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() == 'none':
                    value = None
                else:
                    # Try to convert to number
                    try:
                        if '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        # Keep as string
                        pass
                
                kwargs[key] = value
    
    # Determine whether to save to local
    save_to_local = args.save_to_local and not args.no_save_to_local
    
    # Call load_dataset with parsed arguments
    dataset = DatasetDownLoader.load_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        subset_name=args.subset_name,
        dataset_key_map=dataset_key_map,
        save_to_local=save_to_local,
        local_data_dir=args.local_data_dir,
        cache_dir=args.cache_dir,
        **kwargs
    )
    
    # Print dataset info
    print(f"Successfully loaded dataset: {args.dataset_name}")
    print(f"Dataset type: {type(dataset).__name__}")
    
    # If it's a Dataset or DatasetDict, print more information
    if isinstance(dataset, (Dataset, DatasetDict)):
        if isinstance(dataset, DatasetDict):
            print(f"Dataset splits: {list(dataset.keys())}")
            for split_name in dataset.keys():
                print(f"  {split_name} split has {len(dataset[split_name])} examples")
        else:
            print(f"Dataset has {len(dataset)} examples")
        
        # Print first example if dataset is not empty
        if len(dataset) > 0:
            print("\nFirst example:")
            first_example = dataset[0]
            for key, value in first_example.items():
                # Truncate long values for better display
                if isinstance(value, str) and len(value) > 100:
                    value = value[:97] + "..."
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()





