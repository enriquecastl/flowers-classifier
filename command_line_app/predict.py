from workspace_utils import active_session
from nn_utils import choose_device, process_image
from model import Model
from terminal import get_predict_arguments
import json

def get_category_names(categories, cat_to_names=None):            
    try:
        print(f"Reading category names from {cat_to_names}")
        
        with open(cat_to_names, 'r') as f:
            cat_to_name_map = json.load(f)

            return [cat_to_name_map[str(cat + 1)] for cat in categories]
    except json.JSONDecodeError:
        print(f"{cat_to_names} is not a valid JSON file")
        return None
    except Exception:
        print(f"{cat_to_names} is not a valid file")
        return None

def full_name(categories, categories_names, index):
    full_name_concat = f"{categories[index] + 1} "
    
    if categories_names != None and categories_names[index] != None:
        full_name_concat = f"{full_name_concat}: {categories_names[index]}"
    
    return full_name_concat
    
def display_prediction(ps, categories, categories_names):
    print(f"\nThere is a {ps[0]:.0%} chance that the image is a flower {full_name(categories, categories_names, 0)}\n")    
    print("The other less likely flowers are:\n")
    
    
    for index in range(len(categories)):
        if index == 0:
            continue
            
        name = categories[index]
        probability = ps[index]
                 
        print(f"- {full_name(categories, categories_names, index)}: {ps[index]:.0%}")
    
def main():
    args = get_predict_arguments()
    device = choose_device(args.gpu)
    model = Model.from_checkpoint(args.checkpoint, device)
        
    print(f"Classifying image {args.image} using model with architecture {model.arch}")
    
    with active_session():                    
        ps, categories = model.predict(process_image(args.image), args.top_k)
        
        ps = ps.cpu().numpy().ravel()
        categories = categories.cpu().numpy().ravel()
        categories_names = get_category_names(categories, args.category_names)
        
        display_prediction(ps, categories, categories_names)
        
          
if __name__ == "__main__":
    main()