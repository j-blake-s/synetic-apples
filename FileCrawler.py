import os

from django.core.serializers.json import DjangoJSONEncoder
import json
json.JSONEncoder.default = DjangoJSONEncoder

from tqdm import tqdm

import random


class Directory(object):
  def __init__(self, path, name):
    self._path = path.replace('\\', '/').replace('//', '/')
    self._name = name
    self._files = {}
    self._filesArr = []


class File(object):
  def __init__(self, directory, path, name, extension):
    self._directory = directory
    self._path = path.replace('\\', '/').replace('//', '/')
    self._name = name
    self._extension = extension
    self._pathDir = os.path.dirname(self._path)
    self._dirName = os.path.basename(self._pathDir)
    self._fileSize = None


class FileCrawler(object):
  def __init__(self, p_rootFolder, p_directoryNameContainsFilterSet, p_fileNameContainsFilterSet, p_extensionFilterSet):
    self._rootFolder = p_rootFolder

    self._directoryNameContainsFilterSet = None
    if p_directoryNameContainsFilterSet is not None and len(p_directoryNameContainsFilterSet) > 0:
      self._directoryNameContainsFilterSet = p_directoryNameContainsFilterSet

    self._fileNameContainsFilterSet = None
    if p_fileNameContainsFilterSet is not None and len(p_fileNameContainsFilterSet) > 0:
      self._fileNameContainsFilterSet = p_fileNameContainsFilterSet

    self._extensionFilterSet = None
    if p_extensionFilterSet is not None and len(p_extensionFilterSet) > 0:
      self._extensionFilterSet = p_extensionFilterSet

    self._pathFiles = []

    self._directories = {}
    self._files = {}
    self._filesArr = []

    self.Crawl(p_rootFolder)

    self._lenDirectories = len(self._directories)
    self._lenFiles = len(self._files)
    self._pathFiles = sorted(self._pathFiles)


  def Decimate(self, decimateCount=None, isRandomSample=True):
    files = None
    if decimateCount is not None:
      if isRandomSample:
        files = random.sample(self._filesArr, decimateCount)
      else:
        files = self._filesArr[:decimateCount]
    else:
      files = self._files.values()

    namesExtensionsFiles = {}
    for fileIdx, file in tqdm(enumerate(files), desc="Decimate"):
      namesExtensionsFiles[file._name + file._extension] = file

    return namesExtensionsFiles


  def Crawl(self, p_rootFolder, p_indent=''):
    fname = p_rootFolder.split(os.sep)[-1]
    rootLevelCount = p_rootFolder.count(os.sep)
    for root, dirNames, fileNames in os.walk(p_rootFolder):      
      levelCount = root.count(os.sep) - rootLevelCount
      indent = p_indent + ' ' * (levelCount*2)
      for fileName in fileNames:        
        fF, fFe = os.path.splitext(fileName)
        ff = fileName.lower()
        ff, ffe = os.path.splitext(ff)

        if self._extensionFilterSet is not None:
          if ffe not in self._extensionFilterSet:
            continue

        if self._directoryNameContainsFilterSet is not None:
          directoryNameFilterFound = False
          for directoryNameFilter in self._directoryNameContainsFilterSet:
            if directoryNameFilter in root:
              directoryNameFilterFound = True
              break
          if not directoryNameFilterFound:
            continue

        if self._fileNameContainsFilterSet is not None:
          fileNameFilterFound = False
          for fileNameFilter in self._fileNameContainsFilterSet:
            if fileNameFilter in ff:
              fileNameFilterFound = True
              break
          if not fileNameFilterFound:
            continue

        pathFile = os.path.join(root, fileName)
        pathFile = pathFile.replace('\\', '/').replace('//', '/')
        self._pathFiles.append(pathFile)

        root = root.replace('\\', '/').replace('//', '/')
        if root not in self._directories:
          directoryName = os.path.basename(root)
          directory = Directory(root, directoryName)
          self._directories[directory._path] = directory
        else:
          directory = self._directories[root]

        file = File(directory, pathFile, fF, fFe)
        self._files[file._path] = file
        self._filesArr.append(file)
        directory._files[file._path] = file
        directory._filesArr.append(file)

      for dirName in dirNames:
        dirRootFolder = f'{root}/{dirName}'
        self.Crawl(dirRootFolder, indent)


if __name__ == '__main__':
  print('done.')
