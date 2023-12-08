import argparse
import sys
import os
import glob
import shutil

from PIL import Image

#
#  pcm to adpcm converter class
#
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

#
#  bmp to raw converter class
#
class BMPtoRAW:

  def convert(self, output_file, src_image_dir, screen_width, screen_height, view_width, view_height, use_ibit, rotate):

    rc = 0

    frame0 = False

    with open(output_file, "wb") as f:

      bmp_files = sorted(os.listdir(src_image_dir))
      written_frames = 0

      ofs_x = ( screen_width - view_width ) // 2

      for i, bmp_name in enumerate(bmp_files):

        if bmp_name.lower().endswith(".bmp"):

          im = Image.open(src_image_dir + os.sep + bmp_name)

          im_width, im_height = im.size
          if rotate >= 1 and im_width != view_height:
            print("error: bmp width is not same as view height.")
            return rc
          if rotate == 0 and im_width != view_width:
            print("error: bmp width is not same as view width.")
            return rc

          im_bytes = im.tobytes()

          if rotate >= 1:

            if screen_width == 384 or screen_width == 512:
              grm_bytes = bytearray(512 * im_width * 2)
              for y in range(im_width):
                for x in range(im_height):
                  if rotate == 1:
                    r = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 0 ] >> 3
                    g = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 1 ] >> 3
                    b = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 2 ] >> 3
                  else:
                    r = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 0 ] >> 3
                    g = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 1 ] >> 3
                    b = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 2 ] >> 3
                  c = (g << 11) | (r << 6) | (b << 1)
                  if use_ibit:
                    if rotate == 1:
                      ge = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 1 ] % 8
                    else:
                      ge = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 1 ] % 8
                    if ge >= 4:
                      c += 1
                  else:
                    if c > 0:
                      c += 1
                  grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 0 ] = c // 256
                  grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 1 ] = c % 256
              f.write(grm_bytes)
              written_frames += 1
              print(".", end="", flush=True)

            else:

              if frame0 is False:
                grm_bytes = bytearray(256 * im_width * 2 * 2)
                for y in range(im_width):
                  for x in range(im_height):
                    if rotate == 1:
                      r = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 0 ] >> 3
                      g = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 1 ] >> 3
                      b = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 2 ] >> 3
                    else:
                      r = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 0 ] >> 3
                      g = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 1 ] >> 3
                      b = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 2 ] >> 3
                    c = (g << 11) | (r << 6) | (b << 1)
                    if use_ibit:
                      if rotate == 1:
                        ge = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 1 ] % 8
                      else:
                        ge = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 1 ] % 8
                      if ge >= 4:
                        c += 1
                    else:
                      if c > 0:
                        c += 1
                    grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 0 ] = c // 256
                    grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 1 ] = c % 256
                frame0 = True
              else:
                for y in range(im_width):
                  for x in range(im_height):
                    if rotate == 1:
                      r = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 0 ] >> 3
                      g = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 1 ] >> 3
                      b = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 2 ] >> 3
                    else:
                      r = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 0 ] >> 3
                      g = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 1 ] >> 3
                      b = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 2 ] >> 3
                    c = (g << 11) | (r << 6) | (b << 1)
                    if use_ibit:
                      if rotate == 1:
                        ge = im_bytes[ ((im_height - 1 - x) * im_width + y) * 3 + 1 ] % 8
                      else:
                        ge = im_bytes[ (x * im_width + im_width - 1 - y) * 3 + 1 ] % 8
                      if ge >= 4:
                        c += 1
                    else:
                      if c > 0:
                        c += 1
                    grm_bytes[ y * 512 * 2 + 256 * 2 + (ofs_x + x) * 2 + 0 ] = c // 256
                    grm_bytes[ y * 512 * 2 + 256 * 2 + (ofs_x + x) * 2 + 1 ] = c % 256
                f.write(grm_bytes)
                frame0 = False
                written_frames += 2
                print("..", end="", flush=True)

          else:

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
                  grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 0 ] = c // 256
                  grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 1 ] = c % 256
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
                    grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 0 ] = c // 256
                    grm_bytes[ y * 512 * 2 + (ofs_x + x) * 2 + 1 ] = c % 256
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
                    grm_bytes[ y * 512 * 2 + 256 * 2 + (ofs_x + x) * 2 + 0 ] = c // 256
                    grm_bytes[ y * 512 * 2 + 256 * 2 + (ofs_x + x) * 2 + 1 ] = c % 256
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

#
#  fps class
#
class FPS:

  fps_detail_256 = {
    2: 1.849,
    3: 2.773,
    4: 3.697,
    5: 4.622,
    6: 5.546,
    10: 9.243,
    12: 11.092,
    15: 13.865,
    20: 18.486,
    30: 27.729,
    60: 55.458,
  }

  fps_detail_384 = {
     2: 1.876,
     3: 2.814,
     4: 3.751,
     5: 4.689,
     6: 5.627,
    10: 9.379,
    12: 11.254,
    15: 14.068,
    20: 18.757,
    24: 22.509,
    30: 28.136,
    60: 56.272,
  }

  def get_fps_detail(self, screen_width, fps):
    if screen_width == 384:
      return FPS.fps_detail_384[fps]
    else:
      return FPS.fps_detail_256[fps]


#
#  stage 1 mov to adpcm/pcm
#
def stage1(src_file, src_cut_ofs, src_cut_len, \
           pcm_volume, pcm_peak_max, pcm_avg_min, pcm_freq, pcm_data_file, adpcm_freq, adpcm_wip_file, adpcm_data_file):

  print("[STAGE 1] started.")

  opt = f"-y -i {src_file} " + \
        f"-f s16be -acodec pcm_s16be -filter:a \"volume={pcm_volume},lowpass=f={adpcm_freq}\" -ar {adpcm_freq} -ac 1 -ss {src_cut_ofs} -t {src_cut_len} {adpcm_wip_file} "

  if pcm_freq:
    opt += f"-f s16be -acodec pcm_s16be -filter:a \"volume={pcm_volume}\" -ar {pcm_freq} -ac 2 -ss {src_cut_ofs} -t {src_cut_len} {pcm_data_file}  " \
  
  if os.system(f"ffmpeg {opt}") != 0:
    print("error: ffmpeg failed.")
    return 1
  
  if ADPCM().convert_pcm_to_adpcm(adpcm_wip_file, adpcm_freq, 1, adpcm_data_file, adpcm_freq, pcm_peak_max, pcm_avg_min) != 0:
    print("error: adpcm conversion failed.")
    return 1

  os.remove(adpcm_wip_file) 
  
  print("[STAGE 1] completed.")

  return 0

#
#  stage2 mov to bmp
#
def stage2(src_file, src_cut_ofs, src_cut_len, fps_detail, screen_width, view_width, view_height, deband, sharpness, rotate, output_bmp_dir):

  print("[STAGE 2] started.")

  if view_width is None:
    view_width = screen_width
  elif view_width > screen_width:
    print("error: view_width is too large.")
    return 1

  if rotate >= 1:
    view_width, view_height = view_height, view_width

  os.makedirs(output_bmp_dir, exist_ok=True)

  for p in glob.glob(f"{output_bmp_dir}{os.sep}*.bmp"):
    if os.path.isfile(p):
      os.remove(p)
  
  if sharpness > 0.0:
    sharpness_filter=f",unsharp=3:3:{sharpness}:3:3:0"
  else:
    sharpness_filter=""

  if deband:
    deband_filter=",deband=1thr=0.02:2thr=0.02:3thr=0.02:blur=1"
    deband_filter2="-pix_fmt rgb565"
  else:
    deband_filter=""
    deband_filter2=""

  opt = f"-y -i {src_file} -ss {src_cut_ofs} -t {src_cut_len} " + \
        f"-filter_complex \"[0:v] fps={fps_detail},scale={view_width}:{view_height}{sharpness_filter}{deband_filter}\" " + \
        f"-vcodec bmp {deband_filter2} \"{output_bmp_dir}/output_%05d.bmp\""

  if os.system(f"ffmpeg {opt}") != 0:
    print("error: ffmpeg failed.")
    return 1
  
  print("[STAGE 2] completed.")

  return 0

#
#  stage 3 bmp to raw
#
def stage3(output_bmp_dir, screen_width, view_width, view_height, use_ibit, rotate, raw_data_file):

  print("[STAGE 3] started.")

  if view_width is None:
    view_width = screen_width

  if BMPtoRAW().convert(raw_data_file, output_bmp_dir, screen_width, 256, view_width, view_height, use_ibit, rotate) != 0:
    print("error: BMP to RAW conversion failed.")
    return 1
  
  print("[STAGE 3] completed.")

  return 0

#
#  main
#
def main():

  parser = argparse.ArgumentParser()
  parser.add_argument("src_file", help="source movie file")
  parser.add_argument("rmv_name", help="target rawmv base name")
  parser.add_argument("-fps", help="frame per second", type=int, default=24, choices=[2,3,4,5,6,10,12,15,20,24,30])
  parser.add_argument("-co", "--src_cut_ofs", help="source cut start offset", default="00:00:00.000")
  parser.add_argument("-cl", "--src_cut_len", help="source cut length", default="01:00:00.000")
  parser.add_argument("-sw", "--screen_width", help="screen width", type=int, default=384, choices=[256, 384, 512])
  parser.add_argument("-vw", "--view_width", help="view width", type=int, default=None)
  parser.add_argument("-vh", "--view_height", help="view height", type=int, default=200)
  parser.add_argument("-pv", "--pcm_volume", help="pcm volume", type=float, default=1.0)
  parser.add_argument("-pp", "--pcm_peak_max", help="pcm peak max", type=float, default=98.5)
  parser.add_argument("-pa", "--pcm_avg_min", help="pcm average min", type=float, default=8.0)
  parser.add_argument("-pf", "--pcm_freq", help="16bit pcm frequency", type=int, default=None, choices=[None, 16000, 22050, 24000, 32000, 44100, 48000])
  parser.add_argument("-af", "--adpcm_freq", help="adpcm frequency", type=int, default=15625, choices=[15625, 31250])
  parser.add_argument("-ib", "--use_ibit", help="use i bit for color reduction", action='store_true')
  parser.add_argument("-db", "--deband", help="use debanding filter", action='store_true')
  parser.add_argument("-sp", "--sharpness", help="sharpness (max 1.5)", type=float, default=0.6)
  parser.add_argument("-rt", "--rotate", help="rotate (1:right, 2:left)", type=int, default=0, choices=[0, 1, 2])
  parser.add_argument("-bm", "--preserve_bmp", help="preserve output bmp folder", action='store_true')

  args = parser.parse_args()

  output_bmp_dir = "output_bmp"

  raw_data_file = f"{args.rmv_name}.raw"

  adpcm_wip_file = f"_wip_adpcm.dat"
  adpcm_data_file = f"{args.rmv_name}.p31" if args.adpcm_freq == 31250 else f"{args.rmv_name}.pcm"

  if args.pcm_freq:
    pcm_data_file = f"{args.rmv_name}.s{args.pcm_freq//1000}"
  else:
    pcm_data_file = None

  fps_detail = FPS().get_fps_detail(args.screen_width, args.fps)
  if fps_detail is None:
    print("error: unknown fps")
    return 1

  if stage1(args.src_file, args.src_cut_ofs, args.src_cut_len, \
            args.pcm_volume, args.pcm_peak_max, args.pcm_avg_min, args.pcm_freq, pcm_data_file, \
            args.adpcm_freq, adpcm_wip_file, adpcm_data_file) != 0:
    return 1
  
  if stage2(args.src_file, args.src_cut_ofs, args.src_cut_len, \
            fps_detail, args.screen_width, args.view_width, args.view_height, args.deband, args.sharpness, args.rotate, \
            output_bmp_dir) != 0:
    return 1

  if stage3(output_bmp_dir, args.screen_width, args.view_width, args.view_height, args.use_ibit, args.rotate, raw_data_file):
    return 1

  adpcm_rmv_file = f"{args.rmv_name}_p31.rmv" if args.adpcm_freq == 31250 else f"{args.rmv_name}.rmv"
  colors = 65536 if args.use_ibit else 32768
  with open(adpcm_rmv_file, "w") as f:
    f.write(f"{args.screen_width}\n")
    f.write(f"{args.view_height}\n")
    f.write(f"{args.fps}\n")
    f.write(f"{raw_data_file}\n")
    f.write(f"{adpcm_data_file}\n")
    f.write(f"TITLE:{args.src_file}\n")
    f.write(f"COMMENT:{args.screen_width}x{args.view_height} {colors}colors {fps_detail}fps ADPCM {args.adpcm_freq}Hz mono\n")

  if args.pcm_freq:
    pcm_rmv_file = f"{args.rmv_name}_s{args.pcm_freq//1000}.rmv"
    with open(pcm_rmv_file, "w") as f:
      f.write(f"{args.screen_width}\n")
      f.write(f"{args.view_height}\n")
      f.write(f"{args.fps}\n")
      f.write(f"{raw_data_file}\n")
      f.write(f"{pcm_data_file}\n")
      f.write(f"TITLE:{args.src_file}\n")
      f.write(f"COMMENT:{args.screen_width}x{args.view_height} {colors}colors {fps_detail}fps 16bit PCM {args.pcm_freq}Hz stereo\n")

  if args.preserve_bmp is False:
    shutil.rmtree(output_bmp_dir, ignore_errors=True)

  return 0

if __name__ == "__main__":
  main()
