# Wuji拇指零空间 50类视频审查

观察三点：拇指最后两节是否仍长时间形成生硬直角；物体是否稳定抬升到结尾；退化组是否有明显掌物相对滑移。

## new_transport_gains

- sem-Blender-48bddfc383e475583f36fda0f70067c5[36]：stable=True，transport=True，final=0.352 m，joint4中位=44.9°，近90度=0.0%。
- sem-Book-c8691c86e110318ef2bc9da1ba799c60[11]：stable=True，transport=True，final=0.344 m，joint4中位=84.9°，近90度=0.0%。
- sem-Cookie-ccfa74e5574678325cde8c99e4b182f9[17]：stable=True，transport=True，final=0.342 m，joint4中位=53.6°，近90度=0.0%。
- sem-Ipad-d6a512373ed0cab27298d33994ff64ac[9]：stable=True，transport=True，final=0.345 m，joint4中位=43.9°，近90度=0.0%。
- sem-SoapBar-351dec75ac0619b1827473663798726a[15]：stable=True，transport=True，final=0.327 m，joint4中位=57.7°，近90度=0.0%。

## stable_but_transport_regressions

- core-remote-8f14d5b24d2b798b16a077c4c0fc1181[28]：stable=True，transport=False，final=0.311 m，joint4中位=57.4°，近90度=0.0%。
- sem-Toaster-3ffbb0ab0f8da32f86271197b958e3d5[38]：stable=True，transport=False，final=0.337 m，joint4中位=58.1°，近90度=0.0%。
- sem-WallClock-fc51a35ac0399c457dc65f24011042f1[17]：stable=True，transport=False，final=0.330 m，joint4中位=40.2°，近90度=35.7%。

## worst_thumb_approach

- sem-WineBottle-f331ad8d0e6654ef8f992b1fe7075c8f[10]：stable=True，transport=True，final=0.337 m，joint4中位=62.2°，近90度=35.7%。
- sem-Thumbtack-17aa537d3f70e0998c5d696f6e844329[2]：stable=False，transport=False，final=-0.000 m，joint4中位=51.5°，近90度=25.7%。
- sem-Cap-60104a0945f83b063f30fadefd6911f2[14]：stable=False，transport=False，final=-0.000 m，joint4中位=83.3°，近90度=5.7%。

## representative_clean_success

- sem-Clock-b6d7c946f14d345f7c9f6c5864df1a57[21]：stable=True，transport=True，final=0.328 m，joint4中位=42.4°，近90度=0.0%。
