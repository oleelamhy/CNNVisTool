# -*- mode: python -*-

block_cipher = None


a = Analysis(['CNNVisTool.py'],
             pathex=['c:\\Users\\LH\\Documents\\GitHub\\CNNVisTool'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
a.datas += [('img\\CNNVisTool.ico', 'C:\\Users\\LH\\Nextcloud\\SPECTORS-Object_Recognition\\Software_Developed\\CNNVisTool\\img\\CNNVisTool.ico','DATA')]
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='CNNVisTool_v2.2.0',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True , icon='img\\CNNVisTool.ico')
