from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import PIL.ImageFile
import requests
import time
import sys
import PIL.Image
import os

from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent


import aiohttp
import asyncio
import PIL.Image
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
import io

USER_AGENT = get_datasets_user_agent()

async def fetch_single_image(session, image_url, timeout, retries):
    headers = {"user-agent": USER_AGENT}
    for attempt in range(retries + 1):
        try:
            async with session.get(image_url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    image_data = await response.read()
                    image = PIL.Image.open(io.BytesIO(image_data))
                    return image
        except Exception as e:
            await asyncio.sleep(.5)  # Sleep before retrying
    return None

async def fetch_images(batch, timeout, retries):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_single_image(session, url, timeout, retries)
            for url in batch["image_url"]
        ]
        batch["image"] = await asyncio.gather(*tasks)
    return batch

def fetch_images_map(batch, timeout, retries):
    return asyncio.run(fetch_images(batch, timeout, retries))

if __name__ == '__main__':
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True
    start_time = time.time()
    print("start", start_time)
    split= sys.argv[1]
    dset = load_dataset("google-research-datasets/conceptual_captions", split=split)
    dset = dset.map(fetch_images_map, batched=True, batch_size=100, fn_kwargs={"retries": 1, "timeout": 5})
    filtered_dataset = dset.filter(lambda example: example['image'] is not None)
    print("exisiting images:", filtered_dataset.shape[0]/dset.shape[0])
    filtered_dataset.save_to_disk("/ptmp/agholamzadeh/GCC/ccd_{}.hf".format(split))
    print("done", time.time() - start_time)