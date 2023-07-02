import numpy as np
from os.path import getmtime, exists
from os      import mkdir, getcwd
from numba import njit, cuda
import numba
import re
import itertools
import type_
import subprocess

import uuid


def fn_arg_ano_list( func ):
    result = []
    for arg,ano in func.__annotations__.items():
        if( arg != 'return' ):
            result.append(ano)
    return result

def fn_arg_ano( func ):
    return tuple( x for x in fn_arg_ano_list(func) )
            
def assert_fn_res_ano( func, res_type ):
    if 'return' in func.__annotations__:
        res_ano = func.__annotations__['return']
        if res_ano != res_type :
            arg_str  = str(fn_arg_ano(func))
            ano_str  = arg_str + " -> " + str(res_ano)
            cmp_str  = arg_str + " -> " + str(res_type)
            err_str  = "Annotated function type '" + ano_str               \
            + "' does not match the type deduced by the compiler '"        \
            + cmp_str + "'\nMake sure the definition of the function '"    \
            + func.__name__ + "' results in a return type  matching its "  \
            + "annotation when supplied arguments matching its annotation."
            raise(TypeError(err_str))
            

def global_ptx( func ):
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func))
    assert_fn_res_ano(func, res_type)
    return ptx

def device_ptx( func ):
    ptx, res_type = cuda.compile_ptx_for_current_device(func,fn_arg_ano(func),device=True)
    assert_fn_res_ano(func, res_type)
    return ptx, res_type

def func_defn_time(func):
    return getmtime(func.__globals__['__file__'])




def remove_ptx_comments(ptx_text):
    space = ' \t\r\n'
    ptx_text = "\n".join([ line.lstrip(space) for line in ptx_text.splitlines() if len(line.lstrip(space)) != 0 ])
    
    filtered = []
    start_split = ptx_text.split("/*")
    for index, entry in enumerate(start_split) :
        end_split = entry.split("*/")
        if index == 0 :
            if len(end_split) != 1 :
                raise AssertionError("PTX has an unmatched comment block end")
            filtered.append(end_split[0])
        elif len(end_split) !=2 :
            raise AssertionError("PTX has an unmatched comment block end")
        else :
            filtered.append(end_split[1])

    ptx_text = "".join(filtered)

    ptx_text = "\n".join([ line.lstrip(space) for line in ptx_text.splitlines() if not line.lstrip(space).startswith("//") ])

    ptx_text = "\n".join([ line.split("//")[0] for line in ptx_text.splitlines() ])
    
    return ptx_text

def parse_braks(ptx_text,paren_pairs,closer=None):
    seq = []
    limit = len(ptx_text)
    start = 0
    index = 0
    while index < limit :
        character = ptx_text[index]
        if character == closer :
            seq.append(ptx_text[start:index])
            #print("------------")
            #print(seq)
            return ( closer, seq ), index + 1
        for open,close in paren_pairs :
            if character == open :
                seq.append(ptx_text[start:index])
                index += 1
                sub_seq, delta = parse_braks(ptx_text[index:limit],paren_pairs,close)
                seq.append(sub_seq)
                index += delta
                start = index
        index += 1
    seq.append(ptx_text[start:index])
    #print("------------")
    #print(seq)
    return ( closer, seq ), index
    
                


def parse_sep(ptx_brak_tree, sep):
    closer, seq = ptx_brak_tree
    new_seq = []
    sep_found = False
    sep_seq = []
    seq_len = len(seq)
    for idx, sub_seq in enumerate(seq) :
        if isinstance(sub_seq,str) :
            split_seq = sub_seq.split(sep)
            length = len(split_seq)
            if length == 1 :
                sep_seq.append(split_seq[0])
                #print("\n??????\n")
                #print(sub_seq)
                #print(sep_seq)
            elif length > 1 :
                sep_found = True
                sep_seq.append(split_seq[0])
                new_seq.append((sep,sep_seq))
                sep_seq = [split_seq[-1]]
                for split in split_seq[1:-1]:
                    new_seq.append((sep,[split]))
        elif sub_seq[0] == '}':
            capped = False
            not_last = (idx < seq_len -1)
            incomplete = False
            if len(sep_seq) > 0 and isinstance(sep_seq[0],str):
                incomplete = not sep_seq[0].isspace()
            if incomplete and not_last:
                next = seq[idx+1]
                if isinstance(next,str):
                    split = next.split(sep)
                    empty = len(split[0]) == 0 or split[0].isspace()
                    if len(split) > 1 and empty:
                        capped = True
            if incomplete and not_last and capped:
                sep_seq.append(sub_seq)
                new_seq.append((sep,sep_seq))
                sep_seq = []      
            else:
                new_seq += sep_seq
                sep_seq = []
                new_seq.append(parse_sep(sub_seq,sep))
        else :
            sep_seq.append(parse_sep(sub_seq,sep))
    if sep_found :
        new_seq.append((sep,sep_seq))
        return (closer,new_seq)
    else :
        new_seq += sep_seq
        return (closer,new_seq)


def parse_tok(parse_tree):
    closer, seq = parse_tree
    new_seq = []
    for chunk in seq:
        if isinstance(chunk,str):
            sub_seq = chunk.split()
            if len(sub_seq) > 0:
                new_seq.append((' ',sub_seq))
        else:
            sub_tree = parse_tok(chunk)
            if sub_tree[0] != ' ' or len(sub_tree[1]) != 0:
                new_seq.append(parse_tok(chunk))
    return (closer,new_seq)



def parse_ptx(ptx_text):
    ptx_text = remove_ptx_comments(ptx_text)
    #print(ptx_text)
    braks = [ ('(',')'), ('[',']'), ('{','}'), ('<','>') ]
    parse_tree, _ = parse_braks(ptx_text,braks)
    seperators  = [ ';' , ',', ':' ]
    for sep in seperators:
        parse_tree = parse_sep(parse_tree,sep)
        #print(parse_tree)
    parse_tree = parse_tok(parse_tree)
    #print("\n----\n")
    #print(parse_tree)
    return parse_tree


def delim_match(chunk_list,delim_list):
    if len(delim_list) > len(chunk_list):
        return False
    for index, delim in enumerate(delim_list):
        if chunk_list[index][0] != delim:
            return False
    return True




def extract_regex(parse_tree,regex):
    result = []
    if isinstance(parse_tree,str):
        #for match in re.finditer(regex,parse_tree):
        #    result.append((match, None, None))
        return result

    _, chunks = parse_tree
    for index, chunk in enumerate(chunks):
        if isinstance(chunk,str):
            continue
        sep, content = chunk
        if sep != "ptx":
            result += extract_regex((sep,content),regex)
            continue
        for match in re.finditer(regex,content):
            result.append((match, index, chunks))
    return result


def find_extern_funcs(parse_tree):
    result = []
    extern_regex = r'.extern\s+.func\s*\(\s*.param\s\.+\w+\s+\w+\s*\)\s*(?P<name>\w+)\((?P<params>(\s*.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\)'
    param_regex = r'\s*.param\s+(?P<type>\.\w+)\s+(?P<name>\w+)\s*'
    for match, _, _ in extract_regex(parse_tree,extern_regex):
        params = [(p['type'],p['name']) for p in re.finditer(param_regex,match['params'])]
        result.append((match['name'],params))
    return result


def find_visible_funcs(parse_tree):
    result = []
    extern_regex = r'\.visible\s+\.func\s*\(\s*\.param\s+\.\w+\s+\w+\s*\)\s*(?P<name>\w+)\((?P<params>(\s*\.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\)'
    param_regex = r'\s*\.param\s+(?P<type>\.\w+)\s+(?P<name>\w+)\s*'
    for match, index, context in extract_regex(parse_tree,extern_regex):
        params = [(p['type'],p['name']) for p in re.finditer(param_regex,match['params'])]
        result.append((match['name'],params,context[index+1]))
    return result


def replace(parse_tree,rep_list,whole=False):
    if isinstance(parse_tree,str):
        for targ, dest in rep_list:
            pattern = targ
            if whole:
                pattern = "\\b" + pattern + "\\b"
            parse_tree = re.sub(pattern,dest,parse_tree)
        return parse_tree
    else:
        sep, content = parse_tree
        if sep == "ptx":
            return (sep, replace(content,rep_list,whole))
        new_content = []
        for chunk in content:
            new_content.append(replace(chunk,rep_list,whole))
        return (sep,new_content)
        

def has_curly_block(parse_tree):
    if isinstance(parse_tree,str):
        return False
    sep, content = parse_tree
    if sep == '}':
        return True
    for chunk in content:
        if has_curly_block(chunk):
            return True
    return False

def has_colon(parse_tree):
    if isinstance(parse_tree,str):
        return False
    sep, content = parse_tree
    if sep == ':':
        return True
    for chunk in content:
        if has_colon(chunk):
            return True
    return False

def linify_tree(parse_tree):
    if isinstance(parse_tree,str):
        return parse_tree
    sep, content = parse_tree
    if sep ==';': # and not has_curly_block(parse_tree):
        return  ("ptx",stringify_tree(parse_tree).replace("\n","")+"\n") 

    hit_curly   = False
    hit_line    = False
    line        = []
    new_content = []
    for chunk in content:
        if isinstance(chunk,str):
            new_content.append(chunk)
            continue
        chunk = linify_tree(chunk)
        sub_sep, sub_con = chunk
        if sub_sep == "ptx":
            hit_line = True
            split = sub_con.split(':')
            for index, sub_chunk in enumerate(split):
                #if index != 0:
                #    sub_chunk = sub_chunk[1:]
                if index != len(split)-1 :
                    new_content.append(("ptx",sub_chunk+":\n"))
                else:
                    new_content.append(("ptx",sub_chunk))
        elif sub_sep == '}':
            hit_curly = True
            new_content.append(("ptx",stringify_tree((None,line))+"\n"))
            line = []
            new_content.append((sub_sep,sub_con))
        else:
            line.append((sub_sep,sub_con))
    if len(line) != 0:
        if (hit_curly or hit_line):
            new_content.append(("ptx",stringify_tree((None,line))+"\n"))
        else:
            new_content += line
    return (sep,new_content)


def stringify_tree(parse_tree,inlined=False,last=False,depth=0):
    brak_list = [')',']','}','>']
    brak_map  = {')':'(',']':'[','}':'{','>':'<'}
    sep_list  = [',',';',':']

    tabs = '\t'*depth

    if isinstance(parse_tree,str):
        return " " + parse_tree
    else:
        sep, content = parse_tree

        if sep == "cuda":
            return tabs + content
        
        if sep == "ptx":
            if content.strip() == ";":
                return tabs + '\n' #tabs + ';' + '\n'
            elif inlined:
                return tabs + "asm volatile (\"" + content.rstrip() + "\" ::: \"memory\");\n"
            else:
                return tabs + content
 
        if sep == ' ':
            return "\t".join(content)

        sub_depth = depth
 
        result = ""

        if sep in brak_list:
            if sep == '}':
                sub_depth += 1
                if inlined:
                    result += tabs + "//asm volatile (\"" + str(brak_map[sep]) + "\");\n"
                else:
                    result += tabs + str(brak_map[sep]) + '\n'
            else:
                result += str(brak_map[sep])

        for index, chunk in enumerate(content):
            sub_last = (index == (len(content)-1))
            result += stringify_tree(chunk,inlined,sub_last,sub_depth)

        if sep in brak_list:
            if sep == '}':
                if inlined:
                    result += tabs + "//asm volatile (\"" + str(sep) + "\");\n"
                else:
                    result += tabs + str(sep) + '\n'
            else:
                result += str(sep)

        elif (sep in sep_list) and not last:
            result += str(sep)
            if sep == ';' or sep == ':':
                result += '\n' + tabs
            else:
                result += '\t'
        return result





def strip_ptx(ptx_text,inline):
    ignore_list = [ "//" ]
    if inline:
        ignore_list.extend([".version", ".target" , ".address_size", ".common" ])
    # get rid of empty lines and leading whitespace
    space = ' \t\r\n'
    ptx_text = "\n".join([ line.lstrip(space) for line in ptx_text.splitlines() if len(line.lstrip(space)) != 0 ])
    # get rid of comments and other ptx lines we don't care about
    for entry in ignore_list:
        ptx_text = "\n".join([ line for line in ptx_text.splitlines() if not line.lstrip(space).startswith(entry) ])
    return ptx_text




def replace_call_params(context,call_idx,ret,signature,params,temp_count):
    kind_map = {
        "u8":"h", "u16":"h", "u32":"r", "u64":"l",
        "s8":"h", "s16":"h", "s32":"r", "s64":"l",
        "b8":"h", "b16":"h", "b32":"r", "b64":"l",
        "f32":"f", "f64":"d",
    }

    params.append(ret)

    for id, param in enumerate(params):
        #print(param)
        decl_regex = r"\.param\s+\.\w+\s+"+param+r"\s*;"
        move_regex = r"st\.param\.(?P<kind>\w+)\s*\[\s*"+param+r"\s*(\+\s*[0-9]+\s*)?\]\s*,\s*(?P<src>\w+)\s*;"
        if param == ret:
            move_regex = r"ld\.param\.(?P<kind>\w+)\s*(?P<dst>\w+)\s*,\s*\[\s*"+param+r"\+0\]\s*;"

        start = min(call_idx+2,len(context))
        found_decl = False
        found_move = False
        for line_idx in range(start,-1,-1):
            if found_decl and found_move:
                break
            if isinstance(context[line_idx],str):
                continue
            if context[line_idx][0] != 'ptx':
                continue
            #print(context[line_idx][1])
            if re.match(decl_regex,context[line_idx][1]) != None:
                found_decl = True
                if param == ret:
                    context[line_idx] = ('cuda', "//"+context[line_idx][1])
                else:
                    if id == 0:
                        line = '{kind} *_param_{id};\n'
                    else:
                        line = '{kind} _param_{id};\n'
                    line = line.format(kind=str(signature[id]),id=temp_count+id)
                    context[line_idx] = ('cuda',line)
                continue
            move_match = re.match(move_regex,context[line_idx][1])
            if move_match != None:
                found_move = True
                kind = move_match['kind']
                if param == ret:
                    dst = move_match['dst']
                    #line = 'asm volatile(\"cvt.{kind}.{kind} {dst}, 0;\");\n'
                    line = 'asm volatile(\"mov.{kind} {dst}, 0;\" ::: "memory" );\n'
                    line = line.format(kind=kind,dst=dst)
                    context[line_idx] = ('cuda',line)
                else:
                    src  = move_match['src']
                    #line = 'asm volatile(\"cvt.{kind}.{kind} %0, {src};\" : \"={kid}\"(_param_{id}) : );\n'
                    line = 'asm volatile(\"mov.{kind} %0, {src};\" : \"={kid}\"(_param_{id}) :: "memory" );\n'
                    line = line.format(kind=kind,src=src,id=temp_count+id,kid=kind_map[kind])
                    context[line_idx] = ('cuda',line)
    return temp_count + len(params)-1


def replace_call( parse_tree, mapping, temp_count, type_map, context=[] ):
    repl_fn, repl_name = mapping
    src_list = []
    ret  = None
    call_regex = r"call\.uni\s*\(\s*(?P<ret>\w+)\s*\)\s*,\s*"+repl_fn.name+r"\s*,\s*\((?P<params>\s*\w+\s*(,\s*\w+\s*)*)\)\s*;\s*"
    param_regex = r'\b(?P<name>\w+)\b'
    for match, index, context in extract_regex(parse_tree,call_regex):
        #print(match)

        ret         = match['ret']
        params      = [p['name'] for p in re.finditer(param_regex,match['params'])]
        signature   = [ repl_fn.sig.return_type ] + [ arg for arg in repl_fn.sig.args ]
        signature   = [ map_type_name(type_map,kind) for kind in signature ]

        old_tc = temp_count
        temp_count = replace_call_params(context,index,ret,signature,params,temp_count)
        arg_str = ",".join([ "_param_"+str(x) for x in range(old_tc+1,temp_count)])

        if signature[0] == "void":
            context[index] = ('cuda', repl_name + "(" + arg_str + ");\n" )
        else:
            context[index] = ('cuda', "*_param_0 = " + repl_name + "(" + arg_str + ");\n" )

    return temp_count




def replace_externs( parse_tree, function_map, type_map ):
    temp_count = 0
    for mapping in function_map.items():
        temp_count = replace_call(parse_tree,mapping,temp_count,type_map)



def replace_fn_params( parse_tree, params, has_return ):
    kind_map = {
        "u8":"h", "u16":"h", "u32":"r", "u64":"l",
        "s8":"h", "s16":"h", "s32":"r", "s64":"l",
        "b8":"h", "b16":"h", "b32":"r", "b64":"l",
                             "f32":"f", "f64":"d",
    }
    param_regex = r"ld\.param\.(?P<kind>\w+)\s+(?P<dst>\w+)\s*,\s*\[(?P<name>\w+)\]\s*;"

    for match, index, context in extract_regex(parse_tree,param_regex):
        kind = match['kind']
        dst  = match['dst']
        name = match['name']
        #print(name)

        if name not in params:
            continue

        line = 'asm volatile (\"cvt.{kind}.{kind} {dst}, %0;\" : : \"{kid}\"({name}) : "memory" );\n'
        line = line.format(kind=kind,dst=dst,name=name,kid=kind_map[kind])
        context[index] = ('cuda',line)
    
    return_regex = r"st\.param\.(?P<kind>\w+)\s*\[func_retval0\+0\]\s*,\s*(?P<src>\w+)\s*;"
    for match, index, context in extract_regex(parse_tree,return_regex):
        #if has_return:
        #    kind = match['kind']
        #    src  = match['src']
        #    line = 'asm volatile (\"cvt.{kind}.{kind} %0, {src}\" : \"={kid}\"(result) : );\n'
        #    line = line.format(kind=kind,src=src,kid=kind_map[kind])
        #    context[index] = ('cuda',line )
        #else:
        context[index] = ('cuda',"//"+context[index][1])



def fix_returns(parse_tree,fn_name):
    return_regex = r"ret\s*;"
    for _, index, context in extract_regex(parse_tree,return_regex):
        context[index] = ('ptx',"bra $RETURN_"+fn_name+";")


def fix_temp_param(parse_tree):
    return_regex = r"temp_param_reg;"
    for _, index, context in extract_regex(parse_tree,return_regex):
        context[index] = ('cuda',"//"+context[index][1])



def extern_device_ptx( func, type_map ):
    arg_types   = fn_arg_ano_list(func)
    ptx_text, res_type = device_ptx(func)
    ptx_text = strip_ptx(ptx_text,False)
    print("ptx_text: "+ptx_text)
    parse_tree = parse_ptx(ptx_text)
    line_tree  = linify_tree(parse_tree)
    if func.__name__ == 'make_work':
        print("parse_tree: "+str(parse_tree))
        print("line_tree: "+str(line_tree))
    extern_regex = r'(?P<before>\.visible\s+\.func\s*\(\s*\.param\s+\.\w+\s+\w+\s*\)\s*)(?P<name>\w+)(?P<after>\((?P<params>(\s*\.param\s+\.\w+\s+\w+\s*)(,\s*.param\s+\.\w+\s+\w+\s*)*)\))'
    for match, index, context in extract_regex(line_tree,extern_regex):
        before = match['before']
        after  = match['after']
        #ptx_text = before + "_" + func.__name__ + after
        context[index] = before + "_" + func.__name__ + after
        print("after: "+context[index])
    print("revised: "+str(line_tree))
    ptx_text = stringify_tree(line_tree,False)
    print(ptx_text)
    ptx_text = ptx_text.replace("call.uni","call")
    #if func.__name__ == 'make_work':
        #exit(1)
    return ptx_text
    
    


def inlined_device_ptx( func, function_map, type_map ):
    arg_types   = fn_arg_ano_list(func)
    ptx_text, res_type = device_ptx(func)
    print(ptx_text)
    parse_tree  = parse_ptx(ptx_text)
    print(stringify_tree(linify_tree(parse_tree)))
    #exit(1)
    line_tree   = linify_tree(parse_tree)
    visible_fns = find_visible_funcs(line_tree)
    #print(stringify_tree(linify_tree(parse_tree)))

    if len(visible_fns) == 0:
        raise AssertionError("PTX for user-defined function contains no '.visible .func'")
    elif len(visible_fns) > 1:
        raise AssertionError("PTX for user-defined function contains multiple '.visible .func'")


    name, params, body = visible_fns[0]

    whole_tf_list = []
    inner_tf_list = [
        ("%","_"+func.__name__+"_"),
        ("\$L__","$L_"+func.__name__+"_"),
        ("__local_depot0","_"+func.__name__+"_local_depot0")
    ]
    par_types     = []

    whole_tf_list.append((name,func.__name__))

    targ_par_names  = fn_arg_ano_list(func)
    #print(targ_par_names)

    for index, param in enumerate(params):
        par_name = param[1]
        whole_tf_list.append((par_name,"fn_param_"+str(index)))

    body = replace(body,whole_tf_list,whole=True)
    body = replace(body,inner_tf_list,whole=False)

    replace_externs(body,function_map,type_map)
 
    has_return = 'return' in func.__annotations__
 
    replace_fn_params(body,["fn_param_"+str(index) for index in range(len(params))], has_return)
    fix_returns(body,func.__name__)
    fix_temp_param(body)
    body[1].append(('ptx',"$RETURN_"+func.__name__+":"))
    cuda_body = stringify_tree(body,False,False,2)

    return cuda_body


def map_type_to_np(kind):
    return numba.np.numpy_support.as_dtype(kind)
    primitives = {
        numba.none    : np.void,
        bool          : np.bool8,
        numba.boolean : np.bool8, 
        numba.uint8   : np.uint8, 
        numba.uint16  : np.uint16, 
        numba.uint32  : np.uint32, 
        numba.uint64  : np.uint64, 
        numba.int8    : np.int8, 
        numba.int16   : np.int16, 
        numba.int32   : np.int32, 
        numba.int64   : np.int64, 
        numba.float32 : np.float32, 
        numba.float64 : np.float64
    }
    if kind in primitives:
        return primitives[kind]
    return kind


def alignment(kind):
    align = 1
    for name,type in kind.members:
        member_align = kind.alignof(name)
        if member_align != None and member_align > align:
            align = member_align
    return align



def map_type_name(type_map,kind,rec_mode=""):
    primitives = {
        numba.none : "void",
        bool       : "bool",
        np.bool8   : "bool", 
        np.uint8   : "uint8_t", 
        np.uint16  : "uint16_t", 
        np.uint32  : "uint32_t", 
        np.uint64  : "uint64_t", 
        np.int8    : "int8_t", 
        np.int16   : "int16_t", 
        np.int32   : "int32_t", 
        np.int64   : "int64_t", 
        np.float32 : "float", 
        np.float64 : "double" 
    }

    #print(numba.types.Literal)
    #print(type(numba.types.Literal))
    #print(type(kind))
    #print(kind)

    if kind in primitives:
        return primitives[kind]
    elif isinstance(kind,numba.types.abstract.Literal):
        return map_type_name(type_map,type(kind._literal_value))
    elif isinstance(kind,numba.types.Integer):
        result = "int"
        if not kind.signed :
            result = "u" + result
        result += str(kind.bitwidth)
        return result + "_t"
    elif isinstance(kind,numba.types.Float):
        return "float" + str(kind.bitwidth)
    elif isinstance(kind,numba.types.Boolean):
        return "bool"
    elif isinstance(kind,numba.types.Record):
        #if kind in type_map:
        #    return type_map[kind] + "*"
        #else:
        size  = kind.size
        align = alignment(kind)
        align = 8
        size  = ((size + (align-1)) // align) * align
        result = "_"+str(size)+"b"+str(align)
        if rec_mode == "ptr":
            result += "*"
        elif rec_mode == "void_ptr":
            result = "void*"
        return result
    elif isinstance(kind,numba.types.npytypes.NestedArray):
        return "void*"
    else:
        raise RuntimeError("Unrecognized type '"+str(kind)+"' with type '"+str(type(kind))+"'")


def arg_count(func):
    return len(fn_arg_ano_list(func))

def func_param_text(func,type_map,rec_mode="",prior=False,clip=0,first_void=False):
    param_list  = fn_arg_ano_list(func)

    param_list = param_list[clip:]
    
    if first_void:
        param_list = param_list[1:]
        param_list  = ["void*"] + [ map_type_name(type_map,kind,rec_mode=rec_mode) for kind in param_list ]
    else:
        param_list  = [ map_type_name(type_map,kind,rec_mode=rec_mode) for kind in param_list ]

    param_text  = ", ".join([ kind+" fn_param_"+str(idx+clip+1) for (idx,kind) in enumerate(param_list)])
    if len(param_list) > 0 and prior:
        param_text = ", " + param_text
    return param_text


def func_arg_text(func,type_map,rec_mode="",prior=False,clip=0):
    param_list  = fn_arg_ano_list(func)
    param_list = param_list[clip:]

    arg_text  = ""
    if len(param_list) > 0 and prior:
        arg_text = ", "
    for (idx,kind) in enumerate(param_list):
        if idx != 0:
            arg_text += ", "
        if isinstance(kind,numba.types.Record):
            if   rec_mode == "deref":
                arg_text += "*"
            elif rec_mode == "adrof":
                arg_text += "&"
            elif rec_mode == "cast_deref":
                arg_text += "*("+map_type_name(type_map,kind,rec_mode="ptr")+")"
        arg_text += "fn_param_"+str(idx+clip+1)
    return arg_text


def harm_template_func(func,template_name,function_map,type_map,inline,base=False):


    return_type = "void"
    if 'return' in func.__annotations__:
        return_type = map_type_name(type_map, func.__annotations__['return'])


    param_text  = func_param_text(func,type_map,prior=True,clip=1,first_void=False)
    arg_text    = func_arg_text  (func,type_map,rec_mode="adrof",prior=True,clip=1)

    if base:
        param_text = ""
        arg_text   = ""


    code = "\ttemplate<typename PROGRAM>\n"   \
	     + "\t__device__ static "  \
         + return_type+" "+template_name+"(PROGRAM prog" + param_text + ") {\n"

    if return_type != "void":
        code += "\t\t"+return_type+"  result;\n"
        code += "\t\t"+return_type+" *fn_param_0 = &result;\n"
    else:
        code += "\t\tint  dummy_void_result = 0;\n"
        code += "\t\tint *fn_param_0 = &dummy_void_result;\n"


    #code += "\tprintf(\"{"+func.__name__+"}\");\n"
    #code += "\tprintf(\"(prog%p)\",&prog);\n"
    #code += "\tprintf(\"{ctx%p}\",&prog._dev_ctx);\n"
    #code += "\tprintf(\"(sta%p)\",prog.device);\n"
    #code += "\t//printf(\"{pre%p}\",preamble(prog.device));\n"

    if inline:
        code += inlined_device_ptx(func,function_map,type_map)
    else:
        code += "\t\t_"+func.__name__+"(fn_param_0, &prog"+arg_text+");\n"

    if return_type != "void":
        code += "\t\treturn result;\n"
    code += "\t}\n"

    return code


def pascal_case(name):
    return name.replace("_", " ").title().replace(" ", "")


def harm_async_func(func, function_map, type_map,inline):
    return_type = "void"
    if 'return' in func.__annotations__:
        return_type = map_type_name(type_map, func.__annotations__['return'])
    func_name = str(func.__name__)

    struct_name = pascal_case(func_name)
    param_list  = fn_arg_ano_list(func)[1:]
    param_list  = [ map_type_name(type_map,kind) for kind in param_list ]
    param_text  = ", ".join(param_list)

    code = "struct " + struct_name + " {\n"                                              \
	     + "\tusing Type = " + return_type + "(*)(" + param_text + ");\n"                \
         + harm_template_func(func,"eval",function_map,type_map,inline)                  \
         + "};\n"

    return code



def record_to_struct(record_kind,record_name,type_map):
    result = "struct " + record_name + " {\n"
    for name, kind in record_kind.members:
        result += "\t" + map_type_name(type_map,kind) + " " + name + ";\n"
    result += "};\n"
    return result


def cuda_adr_of(array):
    return array.__cuda_array_interface__['data'][0]


class Runtime():

    def __init__(self,context,context_state,init_fn,exec_fn):
        self.context       = context
        self.context_ptr   = cuda_adr_of(context)
        self.context_state = context_state
        self.state_ptr     = cuda_adr_of(context_state) + 8
        self.init_fn       = init_fn
        self.exec_fn       = exec_fn

    def init(self,wg_count=1):
        self.init_fn[wg_count,32](self.context_ptr,self.state_ptr)

    def exec(self,cycle_count,wg_count=1):
        self.exec_fn[wg_count,32](self.context_ptr,self.state_ptr,cycle_count)

    def load_state(self):
        return self.context_state.copy_to_host()[0]['state']

    def store_state(self):
        pass



class RuntimeSpec():

    def __init__(
            self,
            # The name of the program specification
            spec_name,
            # A triplet containing types which should be
            # held per-device, per-group, and per-thread
            state_spec,
            # A triplet containing the functions that
            # should be run to initialize/finalize states
            # and to generate work for running programs
            base_fns,
            # A list of python functions that are to be
            # included as asyn functions in the program
            async_fns,
            **kwargs 
        ):

        self.spec_name = spec_name


        # The states tracked per-device, per-group, and per-thread
        self.dev_state, self.grp_state, self.thd_state = state_spec

        # The base functions used to set up states and generate work
        self.init_fn, self.final_fn, self.source_fn = base_fns

        self.async_fns = async_fns
        
        self.meta = kwargs

        if 'function_map' in kwargs:
            self.function_map = kwargs['function_map']
        else:
            self.function_map = {}

        if 'type_map' in kwargs:
            self.type_map = kwargs['type_map']
        else:
            self.type_map = {}

        self.generate_meta()
        self.generate_code()




    def generate_meta(self):

        # The number of work links that work groups hold in their shared memory
        if 'STASH_SIZE' not in self.meta:
            self.meta['STASH_SIZE'] = 8

        # The size of the work groups used by the program. Currently, the only
        # supported size is 32
        if 'GROUP_SIZE' not in self.meta:
            self.meta['GROUP_SIZE'] = 32

        # Finding all input types, each represented as dtypes
        input_list = []
        for async_fn in self.async_fns:
            arg_list = []
            for idx, arg in enumerate(fn_arg_ano_list(async_fn)):
                np_type = map_type_to_np(arg)
                arg_list.append(("param_"+str(idx),np_type))
            input_list.append(np.dtype(arg_list))

        # The maximum size of all input types. This must be known in order to
        # allocate memory spans with a size that can support all async functions
        max_input_size = 0
        for input in input_list:
            max_input_size = max(max_input_size,input.itemsize)
        self.meta['MAX_INP_SIZE'] = max_input_size #type_.particle.itemsize

        # The integer type used by the system to address promises
        self.meta['ADR_TYPE']  = np.uint32

        # The type used to store per-link meta-data, currently used only
        # for debugging purposes
        self.meta['META_TYPE'] = np.uint32

        # The type used to store the per-link discriminant, identifying the
        # async call that corresponds to the link's data
        if 'OP_DISC_TYPE' not in self.meta:
            self.meta['OP_DISC_TYPE'] = np.uint32

        # The type used to represent the union of all potential inputs
        self.meta['UNION_TYPE']       = np.dtype((np.void,self.meta['MAX_INP_SIZE']))

        # The type used to represent the array of promises held by a link
        self.meta['PROMISE_ARR_TYPE'] = np.dtype((self.meta['UNION_TYPE'],self.meta['GROUP_SIZE']))

        # The work link type, the fundamental type used to organize work
        # in the system
        self.meta['WORK_LINK_TYPE']   = np.dtype([
            ('promises',self.meta['PROMISE_ARR_TYPE']),
            ('next',self.meta['ADR_TYPE']),
            ('meta_data',self.meta['META_TYPE']),
            ('id',self.meta['OP_DISC_TYPE']),
        ])

        # The size of the array used to house all work links (measured in links)
        if 'WORK_ARENA_SIZE' not in self.meta:
            self.meta['WORK_ARENA_SIZE'] = 65536

        # The type used as a handle to the arena of work links
        self.meta['WORK_ARENA_TYPE']     = np.dtype([('size',np.intp),('links',np.intp)])

        # The number of independent queues maintained per stack frame and per
        # link pool
        if 'QUEUE_COUNT' not in self.meta:
            self.meta['QUEUE_COUNT']     = 8192 #1024
        # The type of value used to represent links
        self.meta['QUEUE_TYPE']          = np.intp
        # The type used to hold a series of work queues
        self.meta['WORK_POOL_ARR_TYPE']  = np.dtype((
            self.meta['QUEUE_TYPE'],
            (self.meta['QUEUE_COUNT'],)
        ))
        
        # The struct type that wraps a series of work queues
        self.meta['WORK_POOL_TYPE']      = np.dtype([
            ('queues',self.meta['WORK_POOL_ARR_TYPE'])
        ])
        # The type used to represent 
        self.meta['WORK_FRAME_TYPE']     = np.dtype([
            ('children_residents',np.uint32),
            ('pool',self.meta['WORK_POOL_TYPE'])
        ])

        # The size of the stack (only size-one stacks are currently supported)
        self.meta['STACK_SIZE']         = 1
        # The type used to hold a series of frames
        self.meta['FRAME_ARR_TYPE']     = np.dtype((
            self.meta['WORK_FRAME_TYPE'],
            (self.meta['STACK_SIZE'],)
        ))
        # The type used to hold a work stack and it's associated meta-data
        self.meta['WORK_STACK_TYPE']    = np.dtype([
            ('checkout',np.uint32),
            ('status_flags',np.uint32),
            ('depth_live',np.uint32),
            ('frames',self.meta['FRAME_ARR_TYPE'])
        ])

        # The device context type, tracking the arena, how much of the arena has
        # been claimed the 'easy' way, the pool for allocating/deallocating work links
        # in the arena, and the stack for tracking work links that contain outstanding
        # promises waiting for processing
        self.meta['DEV_CTX_TYPE'] = np.dtype([
            ('claim_count',np.intp),
            ('arena',self.meta['WORK_ARENA_TYPE']),
            ('pool' ,np.intp),
            ('stack',np.intp)
        ])

        # Bundled device context pointer / device state pair type
        self.meta['DEV_CTX_STA'] = np.dtype([
            ('context',np.intp),
            ('state',self.dev_state)
        ])

        return self.meta



    def generate_harm_code(
            self,
            # Whether or not the async functions should be inlined through ptx
            # (currently, this feature is NOT supported)
            inline=False
        ):


        # Accumulator for type definitions
        type_defs = ""

        # Convert each record type in the type map into a string defining
        # an equivalent C struct.
        for kind, name in self.type_map.items():
            type_defs += record_to_struct(kind,name,self.type_map)


        
        param_specs = {}
        for func in self.async_fns:
            for kind in fn_arg_ano_list(func):
                if isinstance(kind,numba.types.Record):
                    size      = kind.size
                    align     = alignment(kind)
                    param_specs[(size,align)] = ()
        
        state_kinds = [self.dev_state, self.grp_state, self.thd_state]
        for kind in state_kinds:
                if isinstance(kind,numba.types.Record):
                    size      = kind.size
                    align     = alignment(kind)
                    param_specs[(size,align)] = ()
            

        param_decls = "" 
        element_map = {
            1 : "unsigned char",
            2 : "unsigned short int",
            4 : "unsigned int",
            8 : "unsigned long long int"
        }
        for size, align in param_specs.keys():
            align = 8
            count =  (size + (align-1)) // align
            size  = ((size + (align-1)) // align) * align
            param_decls += "struct _"+str(size)+"b"+str(align) \
                        +" { "+element_map[align]+" data["+str(count)+"]; };\n"   


        #union_struct = numba.from_dtype(np.dtype([ ('buffer',np.dtype((np.uint64,union_size))) ]))

        proto_decls = ""                                                             \
            + "extern \"C\" __device__ int _initialize(void*, void* prog);\n"        \
            + "extern \"C\" __device__ int _finalize  (void*, void* prog);\n"        \
            + "extern \"C\" __device__ int _make_work (bool* result, void* prog);\n"
        
        for func in self.async_fns:
            param_text = func_param_text(func,self.type_map,rec_mode="void_ptr",prior=True,clip=0,first_void=True)
            return_type = "void"
            if 'return' in func.__annotations__:
                return_type = map_type_name(self.type_map, func.__annotations__['return'])
            proto_decls += "extern \"C\" __device__ int _"+func.__name__ \
                        +  "("+return_type+"*"+param_text+");\n"

        # Accumulator for async function definitions
        async_defs = ""

        # Forward-declare the structs used to house the definitons
        for func in self.async_fns:
            async_defs += "struct " + pascal_case(func.__name__) + ";\n"

        # Generate an async function struct (with appropriate members)
        # for each async function provided, either attempting to inline the
        # function definition (this currently does not work), or inserting
        # a call to a matching extern function, allowing the scheduling
        # program to trampoline to the appropriate logic
        for func in self.async_fns:
            async_defs += harm_async_func(func,self.function_map,self.type_map,inline)

        # The metaparameters of the program specification, indicating the sizes
        # of the data structures used
        metaparams = {
            "STASH_SIZE" : self.meta['STASH_SIZE'],
            "FRAME_SIZE" : self.meta['QUEUE_COUNT'],
            "POOL_SIZE"  : self.meta['QUEUE_COUNT'],
        }

        # The declarations of each of the metaparameters
        meta_defs = "".join(["\tstatic const size_t "+name+" = "+str(value) +";\n" for name,value in metaparams.items()])

        # The declaration of the union of all input types
        union_def = "\ttypedef OpUnion<"+",".join([pascal_case(str(func.__name__)) for func in self.async_fns])+"> OpSet;\n"

        # The definition of the device, group, and thread states
        state_defs = "\ttypedef "+map_type_name(self.type_map,self.dev_state,rec_mode="ptr")+" DeviceState;\n" \
                   + "\ttypedef "+map_type_name(self.type_map,self.grp_state,rec_mode="ptr")+" GroupState;\n"  \
                   + "\ttypedef "+map_type_name(self.type_map,self.thd_state,rec_mode="ptr")+" ThreadState;\n"

        # The base function definitions
        spec_def = "struct " + self.spec_name + "{\n"                                                     \
                 + meta_defs + union_def + state_defs                                                     \
                 + harm_template_func(self.init_fn  ,"initialize",self.function_map,self.type_map,inline,True) \
                 + harm_template_func(self.final_fn ,"finalize"  ,self.function_map,self.type_map,inline,True) \
                 + harm_template_func(self.source_fn,"make_work" ,self.function_map,self.type_map,inline,True) \
                 + "};\n"

        # String template to alias the appropriate specialization to a convenient name
        spec_decl_template = "typedef {kind}Program<{name}> {short_name};\n"

        preamble = ""#                              \
        #"__device__ void* preamble(void* ptr) {\n"\
        #"\treturn ((void**)ptr)[-1];\n"               \
        #"}\n"


        # String template for the program initialization trampoline
        init_template = ""                                              \
        "extern \"C\" __device__ int init_{short_name}(\n"              \
        "\tvoid*,\n"                                                    \
        "\tvoid *_dev_ctx_arg,\n"                                       \
        "\tvoid *device_arg\n"                                          \
        ") {{\n"                                                        \
        "\tauto _dev_ctx = (typename {short_name}::DeviceContext*) _dev_ctx_arg;\n" \
        "\tauto device   = (typename {short_name}::DeviceState) device_arg;\n"     \
        "\t_inner_dev_init<{short_name}>(*_dev_ctx,device);\n"         \
        "\treturn 0;\n"                                                 \
        "}}\n"

        # String template for the program initialization trampoline
        exec_template = ""                                              \
        "extern \"C\" __device__ "                                      \
        "int exec_{short_name}(\n"                                      \
        "\tvoid*,\n"                                                    \
        "\tvoid   *_dev_ctx_arg,\n"                                     \
        "\tvoid   *device_arg,\n"                                       \
        "\tsize_t  cycle_count\n"                                       \
        ") {{\n"                                                        \
        "\tauto _dev_ctx = (typename {short_name}::DeviceContext*) _dev_ctx_arg;\n" \
        "\tauto device   = (typename {short_name}::DeviceState) device_arg;\n"     \
        "\t_inner_dev_exec<{short_name}>(*_dev_ctx,device,cycle_count);\n"         \
        "\treturn 0;\n"                                                 \
        "}}\n"
        #"\tprintf(\"<ctx%p>\",_dev_ctx);\n"\
        #"\tprintf(\"<sta%p>\",device);\n"\

        dispatch_template = ""                                              \
        "extern \"C\" __device__ \n"                                        \
        "int {short_name}_{fn}_{kind}(void*{params}){{\n"                   \
        "\t(({short_name}*)fn_param_1)->template {kind}<{fn_type}>({args});\n"  \
        "\treturn 0;\n"                                                     \
        "}}\n"
        #"\tprintf(\"{{ {fn} trampoline }}\");\n"                            \

        accessor_template = ""                                              \
        "extern \"C\" __device__ \n"                                        \
        "int {short_name}_{field}(void* result, void* prog){{\n"            \
        "\t(*(void**)result) = {prefix}(({short_name}*)prog)->{field};\n"   \
        "\treturn 0;\n"                                                     \
        "}}\n"
        #"\tprintf(\"{{ {field} accessor }}\");\n"                              \
        #"\tprintf(\"{{prog %p}}\",prog);\n"                              \
        #"\tprintf(\"{{field%p}}\",*(void**)result);\n"                              \

        program_fields = [
            (  "device", ""), (   "group", ""), (  "thread", ""),
            ("_dev_ctx","&"), ("_grp_ctx","&"), ("_thd_ctx","&")
        ]

        # Generate specializations and init/exec trampolines for each
        # program type defined by harmonize
        dispatch_defs = ""
        accessor_defs = ""
        for kind, short_kind in [("Harmonize","hrm"), ("Event","evt")]:
            short_name = self.spec_name.lower()+"_"+short_kind
            dispatch_defs += spec_decl_template.format(kind=kind,name=self.spec_name,short_name=short_name)
            dispatch_defs += init_template.format(short_name=short_name,name=self.spec_name,kind=kind)
            dispatch_defs += exec_template.format(short_name=short_name,name=self.spec_name,kind=kind)
            for (field,prefix) in program_fields:
                accessor_defs += accessor_template.format(short_name=short_name,field=field,prefix=prefix)
            # Generate the dispatch functions for each async function
            for fn in self.async_fns:
                param_text = func_param_text(func,self.type_map,rec_mode="void_ptr",prior=True,clip=0,first_void=True)
                arg_text   = func_arg_text(func,self.type_map,rec_mode="cast_deref",prior=False,clip=1)
                for kind in ["async","sync"]:
                    dispatch_defs += dispatch_template.format(
                        short_name=short_name,
                        fn=fn.__name__,
                        fn_type=pascal_case(fn.__name__),
                        params=param_text,
                        args=arg_text,
                        kind=kind,
                    )


        return preamble + type_defs + param_decls + proto_decls + async_defs + spec_def + dispatch_defs + accessor_defs


    def generate_code(self):

        cache_path = "__ptxcache__/"

        harm_code = self.generate_harm_code()
        harm_filename = cache_path+self.spec_name
        harm_file = open(harm_filename+".cu",mode='w')
        harm_file.write(harm_code)
        harm_file.close()

        comp_cmd = "nvcc "+harm_filename+".cu -arch=native -rdc true " \
                   "-c -include ../harmonize/code/harmonize.cpp -ptx " \
                   "-o "+harm_filename+".ptx"
        subprocess.run(comp_cmd.split(),shell=False,check=True)


        comp_list = [fn for fn in base_fns] + self.async_fns
        self.link_list = [ harm_filename+".ptx" ]
        for fn in comp_list:
            file_name = fn.__name__+".ptx"
            if fn in base_fns:
                file_name = self.spec_name + "_" + file_name
            file_name = cache_path + file_name
            ptx_file  = open(file_name,mode='w')
            ptx_file.write(extern_device_ptx(fn,self.type_map))
            self.link_list.append(file_name)
            ptx_file.close()
        
        init_dev_fn  = cuda.declare_device("init_"+self.spec_name+"_hrm",'void(voidptr,voidptr)')
        exec_dev_fn  = cuda.declare_device("exec_"+self.spec_name+"_hrm",'void(voidptr,voidptr,uintp)')

        vp = numba.types.voidptr

        def init_kernel(context : vp, state : vp):
            init_dev_fn(context,state)

        def exec_kernel(context : vp, state : vp, cycle_count : numba.uintp):
            exec_dev_fn(context,state,cycle_count)

        

        self.init_dispatcher = cuda.jit(
            init_kernel,
            device=False,
            link=self.link_list,
            debug=True,
            opt=False,
            cache=False
        )
        self.exec_dispatcher = cuda.jit(
            exec_kernel,
            device=False,
            link=self.link_list,
            debug=True,
            opt=False,
            cache=False
        )



    def instance(self):

        # The integer in global memory used to track how many 'easy' work link
        # allocations have been made thusfar
        claim_count        = numba.cuda.device_array((1,),dtype=np.uint32)

        # The array that houses all of the work links served by the built-in allocator
        work_arena_array   = numba.cuda.device_array(
            (self.meta['WORK_ARENA_SIZE'],),
            dtype=self.meta['WORK_LINK_TYPE'],
            strides=None,
            order='C',
            stream=0
        )

        # The handle used by the context to hold the work arena's address and size
        work_arena_cpu          = np.zeros((1,),dtype=self.meta['WORK_ARENA_TYPE'])
        work_arena_cpu['size']  = self.meta['WORK_ARENA_SIZE']
        work_arena_cpu['links'] = cuda_adr_of(work_arena_array)

        # The pool used to hold unused work links
        spare_pool         = numba.cuda.device_array((1,),dtype=self.meta['WORK_POOL_TYPE'])
        # The frames used to hold occupied work links
        work_stack         = numba.cuda.device_array((1,),dtype=self.meta['WORK_STACK_TYPE'])

        # The device context, itself
        dev_ctx_cpu        = np.zeros((1,),self.meta['DEV_CTX_TYPE'])
        dev_ctx_cpu[0]['claim_count'] = cuda_adr_of(   claim_count)
        dev_ctx_cpu[0]['arena']       = work_arena_cpu
        dev_ctx_cpu[0]['pool']        = cuda_adr_of(    spare_pool)
        dev_ctx_cpu[0]['stack']       = cuda_adr_of(    work_stack)
        dev_ctx_gpu = numba.cuda.to_device(dev_ctx_cpu)
        dev_ctx_ptr = cuda_adr_of(dev_ctx_gpu)

        # The device state, used by the program
        ctx_sta_cpu = np.zeros((1,),self.meta['DEV_CTX_STA'])
        ctx_sta_cpu[0]['context'] = dev_ctx_ptr
        ctx_sta_gpu = numba.cuda.to_device(ctx_sta_cpu)

        # Pointers to the device contexts and states
        dev_sta_ptr        = cuda_adr_of(ctx_sta_gpu) + 8
        print(hex(dev_sta_ptr))
        print(hex(dev_ctx_ptr))
        #print(hex(cuda_adr_of(work_stack)))
        #print(dev_ctx_cpu)
        print(hex(dev_ctx_cpu[0]['claim_count']))
        print(hex(dev_ctx_cpu[0]['arena']['size']))
        print(hex(dev_ctx_cpu[0]['arena']['links']))
        print(hex(dev_ctx_cpu[0]['pool']))
        print(hex(dev_ctx_cpu[0]['stack']))

        return Runtime(
            dev_ctx_gpu, ctx_sta_gpu,
            self.init_dispatcher,
            self.exec_dispatcher
        )




val_count = 1024
dev_state_type = numba.from_dtype(np.dtype([ ('before', np.uint64), ('val',np.dtype((np.uintp,val_count+1))), ('extra',np.uint32)]))
grp_state_type = numba.from_dtype(np.dtype([ ]))
thd_state_type = numba.from_dtype(np.dtype([ ]))


collaz_iter_dtype = np.dtype([('value',np.int32), ('start',np.int32), ('steps',np.int32)])
collaz_iter = numba.from_dtype(collaz_iter_dtype)

sig = numba.types.void(numba.uintp,collaz_iter)

dev_sig = dev_state_type(numba.uintp)

async_even = cuda.declare_device('collaz_hrm_even_async', sig)
async_odd  = cuda.declare_device('collaz_hrm_odd_async' , sig)
device     = cuda.declare_device('collaz_hrm_device' , dev_sig)


def even(prog: numba.uintp, iter: collaz_iter):
    #numba.cuda.atomic.max(device(prog)['val'],iter['start'],iter['value'])
    iter['steps'] += 1
    iter['value'] /= 2
    if iter['value'] % 2 == 0:
        async_even(prog,iter)
    else :
        async_odd(prog,iter)

@numba.jit(nopython=True)
def immediate_return(param: dev_state_type) -> numba.types.voidptr:
    result : numba.types.voidptr = param
    return result

@numba.jit(nopython=True)
def from_void(param: dev_state_type) -> numba.types.voidptr:
    result : numba.types.voidptr = param
    return result

def odd(prog: numba.uintp, iter: collaz_iter):
    #numba.cuda.atomic.add(device(prog)['val'],iter['start'],1)
    #numba.cuda.atomic.max(device(prog)['val'],iter['start'],iter['steps'])
    
    if iter['value'] <= 1:
        device(prog)['val'][1+iter['start']] = iter['steps']
    else:
        iter['value'] = iter['value'] * 3 + 1
        iter['steps'] += 1
        if iter['value'] % 2 == 0:
            async_even(prog,iter)
        else :
            async_odd(prog,iter)


def initialize(prog: numba.uintp):
    pass

def finalize(prog: numba.uintp):
    pass

def make_work(prog: numba.uintp) -> numba.boolean:
    old = numba.cuda.atomic.add(device(prog)['val'],0,1)
    if old >= val_count:
        return False

    #numba.cuda.atomic.add(device(prog)['val'],1+old,old)
    iter = numba.cuda.local.array(1,collaz_iter)
    iter[0]['value'] = old
    iter[0]['start'] = old
    iter[0]['steps'] = 0

    if old % 2 == 0:
        async_even(prog,iter[0])
    else:
        async_odd (prog,iter[0])
    return True


base_fns   = (initialize,finalize,make_work)
state_spec = (dev_state_type,grp_state_type,thd_state_type) 
async_fns  = [odd,even]

collaz_spec = RuntimeSpec("collaz",state_spec,base_fns,async_fns)

collaz_rt = collaz_spec.instance()

collaz_rt.init(1024)

collaz_rt.exec(6553)

state = collaz_rt.load_state()
print(state)

def collaz_check(value):
    step = 0
    while value > 1:
        step += 1
        if value % 2 == 0:
            value /= 2
        else :
            value = value * 3 + 1
    return step

total = 0
for val in range(val_count):
    steps = collaz_check(val)
    total += steps
    print(steps,end=", ")
print("")
print(f"Total: {total}")

exit(0)


float64 = np.float64
int64   = np.int64
uint64  = np.uint64
bool_   = np.bool_

particle = numba.from_dtype(np.dtype([('x',    float64), ('ux',    float64), ('w',     float64),
                     ('seed', int64  ), ('event', int64  ), ('alive', bool_  ) ]))


external = cuda.declare_device('external', 'int32(float32)')

direct   = cuda.declare_device('direct', 'int32(float32)')


code = harmonize("ProgramSpec",function_map,type_map,state_spec,base_fns,async_fns)
code_file = open("prog.cu",mode='w')
code_file.write(code)
#harmonize("ProgramSpec",function_map,type_map,state_spec,base_fns,async_fns)



PSIZE = 8



np.dtype((np.void, PSIZE))
arena_    = np.dtype([np.uintp,np.uintp])
dev_ctx   = np.dtype([np.uintp,np.uintp])
dev_state = np.dtype([])
init = cuda.declare_device('init_programspecharmonize','void('+dev_ctx+','+dev_state+')')

@cuda.jit(device=False,link=["./prog.cu"],debug=True,opt=False)
def try_entry():
    init()



if False :

    print(ext_func_adapt(simple))


    simple_inner = cuda.declare_device('simple_kernel', 'void()')

    @cuda.jit(device=False, link=["harm.ptx","./__ptxcache__/hrm_simple_dev.ptx"],debug=True,opt=False)
    def do_simple():
        simple_inner()


    do_simple[1,32]()

    #print(harm_func_trampoline(simple))

    print(ext_func_adapt(complicated))
    #print(harm_func_trampoline(complicated))

    #print(particle)



    #print(simple.__annotations__)
    #print(device_ptx(simple))

