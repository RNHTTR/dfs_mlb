def is_str(obj):
      try:
        str(value)
        return True
      except ValueError:
        return False


def is_int(value):
  try:
    int(value)
    return True
  except ValueError:
    return False
