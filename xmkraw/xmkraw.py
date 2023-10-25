import argparse
import sys
import os
import glob
import re

from PIL import Image


class ADPCM:

  step_adjust = [ -1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8 ]

  step_size = [  16,  17,  19,  21,  23,  25,  28,  31,  34,  37,  41,  45,   50,   55,   60,   66,
                 73,  80,  88,  97, 107, 118, 130, 143, 157, 173, 190, 209,  230,  253,  279,  307,
                337, 371, 408, 449, 494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552 ]


  def decode_adpcm(self, code, step_index, last_data):

    ss = ADPCM.step_size[ step_index ]

    delta = ( ss >> 3 )

    if code & 0x01:
      delta += ( ss >> 2 )

    if code & 0x02:
      delta += ( ss >> 1 )

    if code & 0x04:
      delta += ss

    if code & 0x08:
      delta = -delta
      
    estimate = last_data + delta

    if estimate > 2047:
      estimate = 2047

    if estimate < -2048:
      estimate = -2048

    step_index += ADPCM.step_adjust[ code ]

    if step_index < 0:
      step_index = 0

    if step_index > 48:
      step_index = 48

    return (estimate, step_index)


  def encode_adpcm(self, current_data, last_estimate, step_index):

    ss = ADPCM.step_size[ step_index ]

    delta = current_data - last_estimate

    code = 0x00
    if delta < 0:
      code = 0x08         # bit3 = 1
      delta = -delta

    if delta >= ss:
      code += 0x04        # bit2 = 1
      delta -= ss

    if delta >= ( ss >> 1 ):
      code += 0x02        # bit1 = 1
      delta -= ss>>1

    if delta >= ( ss >> 2 ):
      code += 0x01        # bit0 = 1
      
    # need to use decoder to estimate
    (estimate, adjusted_index) = self.decode_adpcm(code, step_index, last_estimate)

    return (code,estimate, adjusted_index)


  def convert_pcm_to_adpcm(self, pcm_file, pcm_freq, pcm_channels, adpcm_file, adpcm_freq, max_peak, min_avg):

    rc = 1

    with open(pcm_file, "rb") as pf:

      pcm_bytes = pf.read()
      pcm_data = []

      pcm_peak = 0
      pcm_total = 0.0
      num_samples = 0

      resample_counter = 0

      if pcm_channels == 2:
        for i in range(len(pcm_bytes) // 4):
          resample_counter += adpcm_freq
          if resample_counter >= pcm_freq:
            lch = int.from_bytes(pcm_bytes[i*4+0:i*4+2], 'big', signed=True)
            rch = int.from_bytes(pcm_bytes[i*4+2:i*4+4], 'big', signed=True)
            pcm_data.append((lch + rch) // 2)
            resample_counter -= pcm_freq
            if abs(lch) > pcm_peak:
              pcm_peak = abs(lch)
            if abs(rch) > pcm_peak:
              pcm_peak = abs(rch)
            pcm_total += float(abs(lch) + abs(rch))
            num_samples += 2
      else:
        for i in range(len(pcm_bytes) // 2):
          resample_counter += adpcm_freq
          if resample_counter >= pcm_freq:
            mch = int.from_bytes(pcm_bytes[i*2+0:i*2+2], 'big', signed=True)
            pcm_data.append(mch)
            resample_counter -= pcm_freq
            if abs(mch) > pcm_peak:
              pcm_peak = abs(mch)
            pcm_total += float(abs(mch))
            num_samples += 1

      avg_level = 100.0 * pcm_total / num_samples / 32767.0
      peak_level = 100.0 * pcm_peak / 32767.0
      print(f"Average Level ... {avg_level:.2f}%")
      print(f"Peak Level    ... {peak_level:.2f}%")

      if avg_level < float(min_avg) or peak_level > float(max_peak):
        print("Level range error. Adjust volume settings.")
        return 1

      last_estimate = 0
      step_index = 0
      adpcm_data = []

      for i,x in enumerate(pcm_data):

        # signed 16bit to 12bit, then encode to ADPCM
        xx = x // 16
        (code, estimate, adjusted_index) = self.encode_adpcm(xx, last_estimate, step_index) 

        # fill a byte in this order: lower 4 bit -> upper 4 bit
        if i % 2 == 0:
          adpcm_data.append(code)
        else:
          adpcm_data[-1] |= code << 4

        last_estimate = estimate
        step_index = adjusted_index

      with open(adpcm_file, 'wb') as af:
        af.write(bytes(adpcm_data))

    return 0


class BMPtoRAW:

  def convert(self, screen_width, src_image_dir, use_ibit, output_file):

    rc = 0

    frame0 = False

    with open(output_file, "wb") as f:

      bmp_files = sorted(os.listdir(src_image_dir))
      written_frames = 0

      for i, bmp_name in enumerate(bmp_files):

        if bmp_name.lower().endswith(".bmp"):

          im = Image.open(src_image_dir + os.sep + bmp_name)

          im_width, im_height = im.size
          if im_width != screen_width:
            print("error: bmp width is not same as screen width.")
            return rc

          im_bytes = im.tobytes()

          if screen_width == 384 or screen_width == 512:

            grm_bytes = bytearray(512 * im_height * 2)
            for y in range(im_height):
              for x in range(im_width):
                r = im_bytes[ (y * im_width + x) * 3 + 0 ] >> 3
                g = im_bytes[ (y * im_width + x) * 3 + 1 ] >> 3
                b = im_bytes[ (y * im_width + x) * 3 + 2 ] >> 3
                c = (g << 11) | (r << 6) | (b << 1)
                if use_ibit:
                  #re = im_bytes[ (y * im_width + x) * 3 + 0 ] % 8
                  ge = im_bytes[ (y * im_width + x) * 3 + 1 ] % 8
                  #if re >= 4 and ge >= 4:
                  if ge >= 4:
                    c += 1
                else:
                  if c > 0:
                    c += 1
                grm_bytes[ y * 512 * 2 + x * 2 + 0 ] = c // 256
                grm_bytes[ y * 512 * 2 + x * 2 + 1 ] = c % 256
            f.write(grm_bytes)
            written_frames += 1
            print(".", end="", flush=True)
                
          else:

            if frame0 is False:
              grm_bytes = bytearray(256 * im_height * 2 * 2)
              for y in range(im_height):
                for x in range(im_width):
                  r = im_bytes[ (y * im_width + x) * 3 + 0 ] >> 3
                  g = im_bytes[ (y * im_width + x) * 3 + 1 ] >> 3
                  b = im_bytes[ (y * im_width + x) * 3 + 2 ] >> 3
                  c = (g << 11) | (r << 6) | (b << 1)
                  if use_ibit:
                    #re = im_bytes[ (y * im_width + x) * 3 + 0 ] % 8
                    ge = im_bytes[ (y * im_width + x) * 3 + 1 ] % 8
                    #if re >= 4 and ge >= 4:
                    if ge >= 4:
                      c += 1
                  else:
                    if c > 0:
                      c += 1
                  grm_bytes[ y * 512 * 2 + x * 2 + 0 ] = c // 256
                  grm_bytes[ y * 512 * 2 + x * 2 + 1 ] = c % 256
              frame0 = True
            else:
              for y in range(im_height):
                for x in range(im_width):
                  r = im_bytes[ (y * im_width + x) * 3 + 0 ] >> 3
                  g = im_bytes[ (y * im_width + x) * 3 + 1 ] >> 3
                  b = im_bytes[ (y * im_width + x) * 3 + 2 ] >> 3
                  c = (g << 11) | (r << 6) | (b << 1)
                  if use_ibit:
                    #re = im_bytes[ (y * im_width + x) * 3 + 0 ] % 8
                    ge = im_bytes[ (y * im_width + x) * 3 + 1 ] % 8
                    #if re >= 4 and ge >= 4:
                    if ge >= 4:
                      c += 1
                  else:
                    if c > 0:
                      c += 1
                  grm_bytes[ y * 512 * 2 + 256 * 2 + x * 2 + 0 ] = c // 256
                  grm_bytes[ y * 512 * 2 + 256 * 2 + x * 2 + 1 ] = c % 256
              f.write(grm_bytes)
              frame0 = False
              written_frames += 2
              print("..", end="", flush=True)

      # drain
      if screen_width == 256 and frame0 and written_frames == len(bmp_files) - 1:
        f.write(grm_bytes)
        frame0 = False
        written_frames += 1
        print(".", end="", flush=True)

      if written_frames == len(bmp_files):
        rc = 0

    print()

    return rc


class FPS:

  fps_detail_256 = {
    10: 9.243,
    12: 11.092,
    15: 13.865,
    20: 18.486,
    24: 22.183,
    30: 27.729,
  }

  fps_detail_384 = {
    10: 9.379,
    12: 11.254,
    15: 14.068,
    20: 18.757,
    24: 22.509,
    30: 28.136,
  }

  def get_fps_detail(self, screen_width, fps):
    if screen_width == 384:
      return FPS.fps_detail_384[fps]
    else:
      return FPS.fps_detail_256[fps]


def stage1(rmv_name, src_file, src_cut_ss, src_cut_to, src_cut_ofs, src_cut_len, pcm_volume, pcm_freq, pcm_file, pcm_file2, adpcm_freq, adpcm_file):

  print("[STAGE 1] started.")

  opt = f"-y -ss {src_cut_ss} -to {src_cut_to} -i {src_file} " + \
        f"-f s16be -acodec pcm_s16be -filter:a 'volume={pcm_volume/100.0}' -ar {pcm_freq}   -ac 2 -ss {src_cut_ofs} -t {src_cut_len} {pcm_file}  " \
        f"-f s16be -acodec pcm_s16be -filter:a 'volume={pcm_volume/100.0}' -ar {adpcm_freq} -ac 1 -ss {src_cut_ofs} -t {src_cut_len} {pcm_file2} " 

  if os.system(f"ffmpeg {opt}") != 0:
    print("error: ffmpeg failed.")
    return 1
  
  if ADPCM().convert_pcm_to_adpcm(pcm_file, pcm_freq, 2, adpcm_file, adpcm_freq, 98.0, 8.5) != 0:
    print("error: adpcm conversion failed.")
    return 1

  os.remove(pcm_file2) 
  
  print("[STAGE 1] completed.")

  return 0


def stage2(rmv_name, output_bmp_dir, src_file, src_cut_ss, src_cut_to, src_cut_ofs, src_cut_len, fps_detail, screen_width, view_height, deband):

  print("[STAGE 2] started.")

  os.makedirs(output_bmp_dir, exist_ok=True)

  for p in glob.glob(f"{output_bmp_dir}{os.sep}*.bmp"):
    if os.path.isfile(p):
      os.remove(p)
  
  if deband:
    deband_filter=",deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1"
  else:
    deband_filter=""

  opt = f"-ss {src_cut_ss} -to {src_cut_to} -i {src_file} -ss {src_cut_ofs} -t {src_cut_len} " + \
        f"-filter_complex '[0:v] fps={fps_detail},scale={screen_width}:{view_height}{deband_filter}' " + \
        f"-vcodec bmp -pix_fmt rgb565 '{output_bmp_dir}/output_%05d.bmp'"

  if os.system(f"ffmpeg {opt}") != 0:
    print("error: ffmpeg failed.")
    return 1
  
  print("[STAGE 2] completed.")

  return 0


def stage3(rmv_name, output_bmp_dir, screen_width, use_ibit, raw_file):

  print("[STAGE 3] started.")

  bmp2raw = BMPtoRAW()

  if bmp2raw.convert(screen_width, output_bmp_dir, use_ibit, raw_file) != 0:
    print("error: BMP to RAW conversion failed.")
    return 1
  
  print("[STAGE 3] completed.")

  return 0


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("src_file", help="source movie file")
  parser.add_argument("rmv_name", help="target rawmv base name")
  parser.add_argument("-cs", "--src_cut_ss", help="source cut start timestamp", default="00:00:00.000")
  parser.add_argument("-ct", "--src_cut_to", help="source cut end timestamp", default="00:06:00.000")
  parser.add_argument("-co", "--src_cut_ofs", help="source cut start offset", default="00:00:00.000")
  parser.add_argument("-cl", "--src_cut_len", help="source cut length", default="00:04:59.500")
  parser.add_argument("-fps", help="frame per second", type=int, default=24)
  parser.add_argument("-sw", "--screen_width", help="screen width", type=int, default=384)
  parser.add_argument("-vh", "--view_height", help="view height", type=int, default=200)
  parser.add_argument("-pv", "--pcm_volume", help="pcm volume", type=int, default=100)
  parser.add_argument("-pf", "--pcm_freq", help="pcm frequency", type=int, default=48000)
  parser.add_argument("-af", "--adpcm_freq", help="adpcm frequency", type=int, default=15625)
  parser.add_argument("-ib", "--use_ibit", help="use i bit for color reduction", action='store_true')
  parser.add_argument("-db", "--deband", help="debanding filter", action='store_true')

  args = parser.parse_args()

  output_bmp_dir = "output_bmp"

  pcm_file = f"{args.rmv_name}.s{args.pcm_freq//1000}"
  pcm_file2 = "_wip_pcm2.dat"
  adpcm_file = f"{args.rmv_name}.pcm"
  raw_file = f"{args.rmv_name}.raw"
  rmv_pcm_file = f"{args.rmv_name}_s{args.pcm_freq//1000}.rmv"
  rmv_adpcm_file = f"{args.rmv_name}_pcm.rmv"

  fps_detail = FPS().get_fps_detail(args.screen_width, args.fps)
  if fps_detail is None:
    print("error: unknown fps")
    return 1

  if stage1(args.rmv_name, args.src_file, args.src_cut_ss, args.src_cut_to, args.src_cut_ofs, args.src_cut_len, args.pcm_volume, args.pcm_freq, pcm_file, pcm_file2, args.adpcm_freq, adpcm_file) != 0:
    return 1
  
  if stage2(args.rmv_name, output_bmp_dir, args.src_file, args.src_cut_ss, args.src_cut_to, args.src_cut_ofs, args.src_cut_len, fps_detail, args.screen_width, args.view_height, args.deband) != 0:
    return 1

  if stage3(args.rmv_name, output_bmp_dir, args.screen_width, args.use_ibit, raw_file):
    return 1

  with open(rmv_pcm_file, "w") as f:
    f.write(f"{args.screen_width}\n")
    f.write(f"{args.view_height}\n")
    f.write(f"{args.fps}\n")
    f.write(f"{raw_file}\n")
    f.write(f"{pcm_file}\n")
    f.write(f"TITLE:\n")
    f.write(f"COMMENT:{args.screen_width}x{args.view_height} {fps_detail}fps 16bit PCM {args.pcm_freq}Hz stereo\n")

  with open(rmv_adpcm_file, "w") as f:
    f.write(f"{args.screen_width}\n")
    f.write(f"{args.view_height}\n")
    f.write(f"{args.fps}\n")
    f.write(f"{raw_file}\n")
    f.write(f"{adpcm_file}\n")
    f.write(f"TITLE:\n")
    f.write(f"COMMENT:{args.screen_width}x{args.view_height} {fps_detail}fps ADPCM {args.adpcm_freq}Hz mono\n")

  return 0

if __name__ == "__main__":
  main()
