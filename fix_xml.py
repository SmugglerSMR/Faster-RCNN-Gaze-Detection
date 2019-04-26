import os
import argparse
import warnings

from xml.dom import minidom

path_annot = '/VOC2012/Annotations'

parser = argparse.ArgumentParser()

parser.add_argument('-folder',
                    help='Folder name to fix (VOCdevkit)',
                    required=True,
                    type=str
                    )

args = parser.parse_args()

for filename in os.listdir(args.folder+ path_annot):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(args.folder+ path_annot, filename)
    
    mydoc = minidom.parse(fullname)
    mydoc.toprettyxml(encoding="utf-8")
    items = mydoc.getElementsByTagName('path')

    filename_str, file_extension = os.path.splitext(filename)

    text = os.path.dirname(os.path.abspath(__file__)) +'/'+ args.folder+'/VOC2012/JPEGImages/' + filename_str + '.jpg'
    items[0].firstChild.data = text    
    #print(items[0].firstChild.data)

    xml_file_handle = open(fullname, 'w')    
    xml_file_handle.write(mydoc.toxml())
    declaration = mydoc.toxml()
    #xml_file_handle.write(mydoc.toprettyxml()[(len(declaration)):])
    #mydoc.writexml(xml_file_handle)
    xml_file_handle.close()    