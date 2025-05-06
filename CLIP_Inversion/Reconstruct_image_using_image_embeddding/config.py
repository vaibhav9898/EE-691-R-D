# CONFIG = {
#     "device": "cuda",
#     "model_name": "ViT-B/32",
#     "img_size": 224,
#     "num_steps": 500,
#     "lr": 0.05,
#     "image_path": "images/apple.jpg",
#     "init_method": "noise",  # or 'gray', or 'noise'
#     "save_every": 100,
#     "method": "optimization",
# }



CONFIG = {
    "device": "cuda",
    "model_name": "ViT-B/32",
    "img_size": 224,
    "num_steps": 800,
    "lr": 0.05,
    "image_path": "images/garden.jpg",
    "init_method": "blur",  # options: 'noise', 'gray', 'blur'
    "save_every": 50,
    "method": "optimization",

    # NEW: number of reconstructions to generate (e.g., for diversity)
    "num_variants": 1,

    # NEW: use cosine similarity loss between CLIP embeddings
    "use_cosine_loss": False,
    "cosine_loss_weight": 0.5,  # tune as needed

    # NEW: whether to save a GIF animation of the reconstruction process
    "save_gif": True,
    "gif_duration": 300  # in milliseconds per frame
}


