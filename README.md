# Visuo-Tactile Cross Generation

We can either predict touch from vision or vision from touch, leading to two subtasks: 1) Vision2Touch: Given an image of a local region on the object’s surface, predict the corresponding tactile RGB image that aligns with the visual image patch in both position and orientation; and 2) Touch2Vision: Given a tactile reading on the object’s surface, predict the corresponding local image patch where the contact happens.

## Usage

#### Training & Evaluation

Start the training process, and test the best model on test-set after training:

```sh
# Train VisGel as an example
python main.py --lr 1e-4 --batch_size 64 \
               --model VisGel \
               --src_modality touch --des_modality vision \
               --patience 500 \
               --exp touch2vision
```

Evaluate the best model in *vision_audio_dscmr*:

```sh
# Evaluate VisGel as an example
python main.py --lr 1e-4 --batch_size 64 \
               --model VisGel \
               --src_modality touch --des_modality vision \
               --patience 500 \
               --exp touch2vision \
               --eval
```

#### Add your own model

To train and test your new model on ObjectFolder Cross-Sensory Retrieval Benchmark, you only need to modify several files in *models*, you may follow these simple steps.

1. Create new model directory

   ```sh
   mkdir models/my_model
   ```

2. Design new model

   ```sh
   cd models/my_model
   touch my_model.py
   ```

3. Build the new model and its optimizer

   Add the following code into *models/build.py*:

   ```python
   elif args.model == 'my_model':
       from my_model import my_model
       model = my_model.my_model(args)
       optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   ```

4. Add the new model into the pipeline

   Once the new model is built, it can be trained and evaluated similarly:

   ```sh
   python main.py --lr 1e-4 --batch_size 64 \
                  --model my_model \
                  --src_modality touch --des_modality vision \
                  --exp touch2vision
   ```

## Results on ObjectFolder Visuo-Tactile Cross Generation Benchmark

We choose 50 objects with rich tactile features and reasonable size, and sample 1, 000 visuo-tactile image pairs on each of them. This results in 50 × 1, 000 = 50, 000 image pairs. We conduct both cross-contact and cross-object experiments by respectively splitting the 1, 000 visuo-tactile pairs of each object into train/validation/test = 800/100/100 and splitting the 50 objects into train/validation/test = 40/5/5. The two settings require the model to generalize to new areas or new objects during testing.

#### Cross-Contact Results

<table>
    <tr>
        <td rowspan="2">Vision</td>
        <td colspan="2">Vision->Touch</td>
        <td colspan="2">Touch->Vision</td>
    </tr>
    <tr>
        <td>PSNR</td>
        <td>SSIM</td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>pix2pix</td>
        <td>22.85</td>
        <td>0.71</td>
        <td>9.16</td>
        <td>0.28</td>
    </tr>
    <tr>
        <td>VisGel</td>
        <td>29.60</td>
        <td>0.87</td>
        <td>14.56</td>
        <td>0.61</td>
    </tr>
</table>

#### Cross-Object Results

<table>
    <tr>
        <td rowspan="2">Vision</td>
        <td colspan="2">Vision->Touch</td>
        <td colspan="2">Touch->Vision</td>
    </tr>
    <tr>
        <td>PSNR</td>
        <td>SSIM</td>
        <td>PSNR</td>
        <td>SSIM</td>
    </tr>
    <tr>
        <td>pix2pix</td>
        <td>18.91</td>
        <td>0.63</td>
        <td>7.03</td>
        <td>0.12</td>
    </tr>
    <tr>
        <td>VisGel</td>
        <td>25.91</td>
        <td>0.82</td>
        <td>12.61</td>
        <td>0.38</td>
    </tr>
</table>