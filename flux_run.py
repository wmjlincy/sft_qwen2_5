import torch
from diffusers import FluxPipeline

from transformers import CLIPProcessor, CLIPModel



model_name = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
 
def compute_clip_score(image, text):
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits_per_image.item()


pipe = FluxPipeline.from_pretrained("./FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power



prompts = ["A cat holding a sign that says hello world",  "A driving image in city", 
           'A realistic photo of a Beijing city traffic intersection taken by canon camera: -It is sunny day. - Traffic lights are shown in red. - Old buildings on the side of the road. - There are clouds in the sky. -The traffic sign shows “road people”.  Make sure the picture looks natural and meets all of the above criteria',
         "Scene Description:  \nThe image shows a multi-lane highway during twilight or early evening. The road is relatively clear with moderate traffic. The weather appears to be clear, and the sky has a gradient of colors indicating either sunrise or sunset.\n\nTraffic Signs:  \nThere is a blue directional sign on the right side of the road. It indicates directions for specific destinations but the text is not clearly visible in the image.\n\nVehicle Information:  \nA large white truck is on the left side of the road, moving in the same direction as the viewer's vehicle. In front of the viewer's vehicle, there is a car driving ahead in the same lane. Other vehicles can be seen further down the road, all moving in the same direction.\n\nPedestrian Information:  \nNo pedestrians are visible in the image. However, there is a motorcyclist on the far right side of the road, near the divider.\n\nObstacle Information:  \nThe main obstacles include the large truck on the left and the car directly in front. There are no visible cones or other physical obstacles on the road.\n\nRoad Condition Information:  \nThe road appears to be in good condition with clear lane markings. The lanes are wide, and there is a divider separating the opposing traffic.\n\nPotential Risks:  \nThe presence of the large truck could pose a risk if it suddenly changes lanes. The motorcyclist on the right may also be a potential hazard if they move into the viewer's lane. Additionally, the car in front could brake unexpectedly.\n\nDriving Suggestions:  \nMaintain a safe following distance from the car in front. Be cautious of the large truck on the left and monitor its movements. Keep an eye on the motorcyclist on the right. Adjust speed as necessary and be prepared to change lanes if required for safety." ]

similar_promt = [
    
        'cat, sing',
        'traffic intersection, traffic light',
        'highway, raod'
        ]

i = 0
num = 0
N = 10
while num < N:
    for prompt in prompts:
        image = pipe(
        prompt,
        height=900,
        width=1600,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cuda")
        ).images[0]
        
    
        score = compute_clip_score(image, similar_promt)
        i+=1
        if score > 20:
            image.save(str(num) + ".png")
            num+=1
    
    