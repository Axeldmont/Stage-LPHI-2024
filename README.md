# Mactrack :

## Getting Started : 

First you need to install those packages in order to use the program :
```
pip install kartezio
pip install opencv-python
```
First, you need to define the input folder you will use for the program, your input folder should look like this : 
```
input/yourdatafolder
  ├── dataset
  ├── models
  ├── results
  ├── vert
      ├── frames #(empty)
      └── greenchannelvideo.mp4
  └── redchannelvideo.mp4
```
You can now run the program **gettingstarted.py** who will show each step and function you can use. 

**Warning !** You need to add this class to the file "~/anaconda3/Lib/site-packages/numena/io/json.py" 

```python
class Serializable(ABC):
    """Python Interface to export Python objects to JSON format.

    The child class must implement 'dumps' property.
    """

    @abstractmethod
    def dumps(self) -> dict:
        """Property to generate the JSON formatted data of the current instance."""
        pass
```

