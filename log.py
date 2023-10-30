#  simulator-2022-nca - Simpy simulator of online scheduling between edge nodes
#  Copyright (c) 2021 - 2022. Gabriele Proietti Mattia <pm.gabriele@outlook.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from __future__ import annotations

"""
Implement a colored logging
"""


class Colors:
    """Colors list"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class status_str:
    CHECK_STR = " " + Colors.WARNING + "CHCK" + Colors.ENDC + " "
    OK_STR = "  " + Colors.OKGREEN + "OK" + Colors.ENDC + "  "
    DEAD_STR = " " + Colors.FAIL + "DEAD" + Colors.ENDC + " "
    MISM_STR = " " + Colors.WARNING + "MISM" + Colors.ENDC + " "
    WARN_STR = " " + Colors.WARNING + "WARN" + Colors.ENDC + " "


COLOR = True


class Log(object):

    @staticmethod
    def err(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.FAIL, end="")
            print(value, *args, end="", sep=sep)
            print(Colors.ENDC, end=end)
        else:
            print(value, args, end=end, sep=sep, file=file)

    @staticmethod
    def fatal(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.FAIL, end="")
            print(value, *args, end="", sep=sep)
            print(Colors.ENDC, end=end)
        else:
            print(value, args, end=end, sep=sep, file=file)
        raise RuntimeError(value)

    @staticmethod
    def warn(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.WARNING, end="")
            print(value, *args, end="")
            print(Colors.ENDC)
        else:
            print(value, args, end=end, sep=sep, file=file)

    @staticmethod
    def info(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.OKBLUE, end="")
            print(value, *args, end="")
            print(Colors.ENDC)
        else:
            print(value, args, end=end, sep=sep, file=file)

    @staticmethod
    def debug(value, *args, end="\n", sep="", file=None):
        print(value, *args, end=end, sep=sep, file=file)

    #
    # Module log
    #

    @staticmethod
    def merr(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.err(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def mwarn(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.warn(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def minfo(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.info(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def mdebug(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.debug(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def mfatal(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.fatal(new_value, *args, end=end, sep=sep, file=file)
